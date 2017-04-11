// Copyright ©2017 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iterative

import (
	"errors"
	"math"

	"github.com/gonum/floats"
)

// BiCGSTAB implements the BiConjugate Gradient STABilized iterative method with
// preconditioning for solving the system of linear equations
//  Ax = b,
// where A is a non-symmetric matrix. For symmetric positive definite systems
// use CG.
//
// BiCGSTAB needs MatVec and PSolve matrix operations.
type BiCGSTAB struct {
	first  bool
	resume int

	rho, rhoPrev float64
	alpha        float64
	omega        float64

	rt   []float64
	p    []float64
	v    []float64
	t    []float64
	phat []float64
	s    []float64
	shat []float64
}

// Init implements the Method interface.
func (b *BiCGSTAB) Init(dim int) {
	if dim <= 0 {
		panic("iterative: dimension not positive")
	}

	b.rt = reuse(b.rt, dim)
	b.p = reuse(b.p, dim)
	b.v = reuse(b.v, dim)
	b.t = reuse(b.t, dim)
	b.phat = reuse(b.phat, dim)
	b.s = reuse(b.s, dim)
	b.shat = reuse(b.shat, dim)
	b.first = true
	b.resume = 1
}

// Iterate implements the Method interface.
func (b *BiCGSTAB) Iterate(ctx *Context) (Operation, error) {
	switch b.resume {
	case 1:
		if b.first {
			copy(b.rt, ctx.Residual)
		}
		b.rho = floats.Dot(b.rt, ctx.Residual)
		if b.rho < dlamchE*dlamchE {
			b.resume = 0 // Calling Iterate again without Init will panic.
			return NoOperation, errors.New("iterative: rho breakdown")
		}
		if b.first {
			copy(b.p, ctx.Residual)
		} else {
			beta := (b.rho / b.rhoPrev) * (b.alpha / b.omega)
			floats.AddScaled(b.p, -b.omega, b.v) // p_i -= ω * v_i
			floats.Scale(beta, b.p)              // p_i *= β
			floats.Add(b.p, ctx.Residual)        // p_i += r_i
		}
		ctx.Src = b.p
		ctx.Dst = b.phat
		b.resume = 2
		return PSolve, nil
		// Solve M p^_i = p_i.
	case 2:
		ctx.Src = b.phat
		ctx.Dst = b.v
		b.resume = 3
		return MatVec, nil
		// Compute Ap^_i -> v_i.
	case 3:
		b.alpha = b.rho / floats.Dot(b.rt, b.v)
		// Early check for tolerance.
		floats.AddScaled(ctx.Residual, -b.alpha, b.v)
		copy(b.s, ctx.Residual)
		ctx.Src = nil
		ctx.Dst = nil
		ctx.ResidualNorm = floats.Norm(ctx.Residual, 2)
		ctx.Converged = false
		b.resume = 4
		return CheckResidualNorm, nil
	case 4:
		if ctx.Converged {
			floats.AddScaled(ctx.X, b.alpha, b.phat)
			b.resume = 0 // Calling Iterate again without Init will panic.
			return EndIteration, nil
		}
		ctx.Src = ctx.Residual
		ctx.Dst = b.shat
		b.resume = 5
		return PSolve, nil
		// Solve M s^_i = r_i.
	case 5:
		ctx.Src = b.shat
		ctx.Dst = b.t
		b.resume = 6
		return MatVec, nil
		// Compute As^_i -> t_i.
	case 6:
		b.omega = floats.Dot(b.t, b.s) / floats.Dot(b.t, b.t)
		floats.AddScaled(ctx.X, b.alpha, b.phat)
		floats.AddScaled(ctx.X, b.omega, b.shat)
		floats.AddScaled(ctx.Residual, -b.omega, b.t)
		ctx.Src = nil
		ctx.Dst = nil
		ctx.ResidualNorm = floats.Norm(ctx.Residual, 2)
		ctx.Converged = false
		b.resume = 7
		return CheckResidualNorm, nil
	case 7:
		if ctx.Converged {
			b.resume = 0 // Calling Iterate again without Init will panic.
			return EndIteration, nil
		}
		if math.Abs(b.omega) < dlamchE*dlamchE {
			return NoOperation, errors.New("iterative: omega breakdown")
		}
		b.rhoPrev = b.rho
		b.first = false
		b.resume = 1
		return EndIteration, nil

	default:
		panic("iterative: BiCGSTAB.Init not called")
	}
}
