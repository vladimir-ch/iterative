// Copyright ©2016 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iterative

import "github.com/gonum/floats"

type CG struct {
	first        bool
	rho, rhoPrev float64
	resume       int
}

func (cg *CG) Init(dim int) int {
	cg.first = true
	cg.resume = 1
	return 4
}

func (cg *CG) Iterate(ctx *Context) (Operation, error) {
	const (
		ri = iota
		zi
		pi
		Api
	)
	r := ctx.Vectors[ri]
	switch cg.resume {
	case 1:
		if cg.first {
			copy(r, ctx.Residual)
		}
		ctx.Src = ri
		ctx.Dst = zi
		cg.resume = 2
		return PSolve, nil
		// Solve M z = r_{i-1}
	case 2:
		z, p := ctx.Vectors[zi], ctx.Vectors[pi]
		cg.rho = floats.Dot(r, z) // ρ_i = r_{i-1} · z
		if !cg.first {
			beta := cg.rho / cg.rhoPrev  // β = ρ_i / ρ_{i-1}
			floats.AddScaled(z, beta, p) // z = z + β p_{i-1}
		}
		copy(p, z) // p_i = z

		ctx.Src = pi
		ctx.Dst = Api
		cg.resume = 3
		return MatVec, nil
		// Compute Ap_i
	case 3:
		p, Ap := ctx.Vectors[pi], ctx.Vectors[Api]
		alpha := cg.rho / floats.Dot(p, Ap) // α = ρ_i / (p_i · Ap_i)
		floats.AddScaled(r, -alpha, Ap)     // r_i = r_{i-1} - α Ap_i
		floats.AddScaled(ctx.X, alpha, p)   // x_i = x_{i-1} + α p_i

		copy(ctx.Residual, r)
		ctx.Src = -1
		ctx.Dst = -1
		ctx.Converged = false
		cg.resume = 4
		return CheckResidual, nil
	case 4:
		if ctx.Converged {
			cg.resume = 0
			return EndIteration, nil
		}
		cg.rhoPrev = cg.rho
		cg.first = false
		cg.resume = 1
		return EndIteration, nil

	default:
		panic("iterative: CG.Init not called")
	}
}
