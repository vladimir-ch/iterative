// Copyright ©2016 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iterative

import "github.com/gonum/floats"

type CG struct {
	first     bool
	rho, rho1 float64
	resume    int
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
	r := ctx.Work[ri]
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
		z, p := ctx.Work[zi], ctx.Work[pi]
		cg.rho = floats.Dot(r, z) // ρ_i = r_{i-1} · z
		if cg.first {
			cg.first = false
		} else {
			beta := cg.rho / cg.rho1     // β = ρ_i / ρ_{i-1}
			floats.AddScaled(z, beta, p) // z = z + β p_{i-1}
		}
		copy(p, z) // p_i = z

		ctx.Src = pi
		ctx.Dst = Api
		cg.resume = 3
		return MatVec, nil
		// Compute Ap_i
	case 3:
		p, Ap := ctx.Work[pi], ctx.Work[Api]
		alpha := cg.rho / floats.Dot(p, Ap) // α = ρ_i / (p_i · Ap_i)
		floats.AddScaled(r, -alpha, Ap)     // r_i = r_{i-1} - α Ap_i
		floats.AddScaled(ctx.X, alpha, p)   // x_i = x_{i-1} + α p_i

		copy(ctx.Residual, r)
		ctx.Src = -1
		ctx.Dst = -1
		cg.resume = 4
		return CheckConvergence, nil
	case 4:
		cg.rho1 = cg.rho
		cg.resume = 1
		return Iteration, nil
	}
	panic("unreachable")
}
