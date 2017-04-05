// Copyright ©2017 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iterative

import (
	"math"

	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
	"github.com/gonum/floats"
)

type GMRES struct {
	// Restart is the restart parameter.
	// It must be 0 <= Restart <= dim.
	// If it is 0, it will be set to dim.
	Restart int

	resume int
	i      int // Counter for inner iterations.

	s  []float64
	w  []float64
	y  []float64
	av []float64

	v    []float64
	ldv  int
	h    []float64
	ldh  int
	givs []givens
}

type givens struct {
	c, s float64
}

func (g *GMRES) Init(dim int) {
	if dim <= 0 {
		panic("iterative: invalid dim")
	}

	if g.Restart == 0 {
		g.Restart = dim
	}
	if g.Restart <= 0 || dim < g.Restart {
		panic("iterative: invalid GMRES.Restart")
	}

	g.s = reuse(g.s, dim)
	g.w = reuse(g.w, dim)
	g.y = reuse(g.y, dim)
	g.av = reuse(g.av, dim)

	k := g.Restart
	g.ldv = dim
	g.v = reuse(g.v, g.ldv*(k+1))
	g.ldh = k + 1
	g.h = reuse(g.h, g.ldh*k)
	if cap(g.givs) < k {
		g.givs = make([]givens, k)
	} else {
		g.givs = g.givs[:k]
	}

	g.resume = 1
}

func (g *GMRES) Iterate(ctx *Context) (Operation, error) {
	n := len(ctx.X)
	ldv := g.ldv
	switch g.resume {
	case 1:
		// Construct the first column of V.
		ctx.Src = ctx.Residual
		ctx.Dst = g.v[:n]
		g.resume = 2
		return PSolve, nil
		// Solve M V[:,0] = r.
	case 2:
		// Normalize V[:,0].
		rnorm := floats.Norm(g.v[:n], 2)
		floats.Scale(1/rnorm, g.v[:n])
		// Initialize s to the elementary vector e_1 scaled by rnorm.
		for i := range g.s {
			g.s[i] = 0
		}
		g.s[0] = rnorm

		// for i := 0; i < Restart; i++ {
		g.i = 0
		fallthrough
	case 3:
		i := g.i
		if i == g.Restart {
			g.resume = 7
			ctx.Src = nil
			ctx.Dst = nil
			return NoOperation, nil
		}
		ctx.Src = g.v[i*ldv : i*ldv+n]
		ctx.Dst = g.av
		g.resume = 4
		// Compute A V[:,i].
		return MatVec, nil
	case 4:
		ctx.Src = g.av
		ctx.Dst = g.w
		g.resume = 5
		// Solve M w = A V[:,i].
		return PSolve, nil
	case 5:
		i := g.i
		h := g.h
		ldh := g.ldh

		// Construct i-th column of the upper Hessenberg matrix using
		// the Gram-Schmidt process on V and W so that it is orthonormal
		// to the previous i-1 columns.
		for k := 0; k <= i; k++ {
			vk := g.v[k*ldv : k*ldv+n]
			hki := floats.Dot(vk, g.w)
			h[k+i*ldh] = hki
			floats.AddScaled(g.w, -hki, vk)
		}
		wnorm := floats.Norm(g.w, 2)
		hi := h[i*ldh : i*ldh+g.Restart+1]
		h[i+1] = wnorm // H[i+1,i] = |w|
		vip1 := g.v[(i+1)*ldv : (i+1)*ldv+n]
		copy(vip1, g.w)
		floats.Scale(1/wnorm, vip1)

		// Apply (i-1) Givens rotation matrices to the i-th
		// column of H.
		for j := 0; j < i; j++ {
			hi[j], hi[j+1] = rotvec(hi[j], hi[j+1], g.givs[j])
		}
		// Compute the (i+1)st Givens rotation that zeroes H[i+1,i].
		g.givs[i] = drotg(hi[i], hi[i+1])
		// Apply the (i+1)st Givens rotation.
		hi[i], hi[i+1] = rotvec(hi[i], hi[i+1], g.givs[i])

		// Apply the (i+1)st Givens rotation to (s[i], s[i+1]).
		g.s[i], g.s[i+1] = rotvec(g.s[i], g.s[i+1], g.givs[i])
		// Approximate the residual norm and check for convergence.
		ctx.ResidualNorm = math.Abs(g.s[i+1])
		ctx.Src = nil
		ctx.Dst = nil
		ctx.Converged = false
		g.resume = 6
		return CheckResidualNorm, nil
	case 6:
		if ctx.Converged {
			// Compute final approximate solution x and finish.
			g.update(ctx.X)
			g.resume = 0
			return EndIteration, nil
		}
		g.i++
		g.resume = 3
		return NoOperation, nil
		// end for loop
	case 7:
		// Compute final approximate solution x.
		g.update(ctx.X)
		g.resume = 8
		// Compute final residual.
		return ComputeResidual, nil
	case 8:
		ctx.Converged = false
		g.resume = 9
		// Check for convergence.
		return CheckResidual, nil
	case 9:
		if ctx.Converged {
			g.resume = 0
			return EndIteration, nil
		}
		g.resume = 1
		// TODO: Add IterationLimit bool to Context.
		return EndIteration, nil

	default:
		panic("iterative: GMRES.Init not called")
	}
}

func (g *GMRES) update(x []float64) {
	i := g.i
	y := g.y[:i+1]
	copy(y, g.s[:i+1])
	// Solve H*y = s for upper triangular H.
	// H is upper triangular but stored in column-major order while Dtrsv
	// expects row-major.
	bi := blas64.Implementation()
	bi.Dtrsv(blas.Lower, blas.Trans, blas.NonUnit, g.i+1, g.h, g.Restart+1, g.y[:i+1], 1)
	// Compute current solution vector x.
	n := len(x)
	ldv := g.ldv
	for j := 0; j <= g.i; j++ {
		vj := g.v[j*ldv : j*ldv+n]
		floats.AddScaled(x, y[j], vj)
	}
}

func drotg(a, b float64) givens {
	if b == 0 {
		return givens{c: 1, s: 0}
	}
	if math.Abs(b) > math.Abs(a) {
		tmp := -a / b
		s := 1 / math.Sqrt(1+tmp*tmp)
		return givens{c: tmp * s, s: s}
	}
	tmp := -b / a
	c := 1 / math.Sqrt(1+tmp*tmp)
	return givens{c: c, s: tmp * c}
}

func rotvec(x, y float64, g givens) (rx, ry float64) {
	rx = g.c*x - g.s*y
	ry = g.s*x + g.c*y
	return
}
