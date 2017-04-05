// Copyright ©2017 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iterative

import (
	"errors"
	"math"

	"github.com/gonum/floats"
)

// BiCGStab implements the BiConjugate Gradient Stabilized iterative method with
// preconditioning for solving the system of linear equations
//  Ax = b,
// where A is a non-symmetric matrix. For symmetric positive definite systems
// use CG.
//
// BiCGStab needs MatVec and PSolve matrix operations.
type BiCGStab struct {
	first        bool
	rho, rhoPrev float64
	alpha        float64
	omega        float64
	resume       int
}

// Init implements the Method interface.
func (bicg *BiCGStab) Init(dim int) int {
	bicg.first = true
	bicg.resume = 1
	return 7
}

// Iterate implements the Method interface.
func (bicg *BiCGStab) Iterate(ctx *Context) (Operation, error) {
	const (
		ri    = 0
		rti   = 1
		pi    = 2
		vi    = 3
		ti    = 4
		phati = 5
		shati = 6
		si    = 0
	)
	switch bicg.resume {
	case 1:
		if bicg.first {
			copy(ctx.Vectors[ri], ctx.Residual)
			copy(ctx.Vectors[rti], ctx.Vectors[ri])
		}
		bicg.rho = floats.Dot(ctx.Vectors[rti], ctx.Vectors[ri])
		if bicg.rho < dlamchE*dlamchE {
			bicg.resume = 0
			return NoOperation, errors.New("iterative: rho breakdown")
		}
		if bicg.first {
			copy(ctx.Vectors[pi], ctx.Vectors[ri])
		} else {
			beta := (bicg.rho / bicg.rhoPrev) * (bicg.alpha / bicg.omega)
			floats.AddScaled(ctx.Vectors[pi], -bicg.omega, ctx.Vectors[vi]) // p_i -= ω * v_i
			floats.Scale(beta, ctx.Vectors[pi])                             // p_i *= β
			floats.Add(ctx.Vectors[pi], ctx.Vectors[ri])                    // p_i += r_i
		}
		ctx.Src = pi
		ctx.Dst = phati
		bicg.resume = 2
		return PSolve, nil
		// Solve M p^_i = p_i.
	case 2:
		ctx.Src = phati
		ctx.Dst = vi
		bicg.resume = 3
		return MatVec, nil
		// Compute Ap^_i -> v_i.
	case 3:
		bicg.alpha = bicg.rho / floats.Dot(ctx.Vectors[rti], ctx.Vectors[vi])
		// Early check for tolerance.
		floats.AddScaled(ctx.Vectors[ri], -bicg.alpha, ctx.Vectors[vi])
		copy(ctx.Vectors[si], ctx.Vectors[ri])
		copy(ctx.Residual, ctx.Vectors[si])
		ctx.Src = -1
		ctx.Dst = -1
		ctx.Converged = false
		bicg.resume = 4
		return CheckResidual, nil
	case 4:
		if ctx.Converged {
			floats.AddScaled(ctx.X, bicg.alpha, ctx.Vectors[phati])
			bicg.resume = 0
			return EndIteration, nil
		}
		ctx.Src = ri
		ctx.Dst = shati
		bicg.resume = 5
		return PSolve, nil
		// Solve M s^_i = r_i.
	case 5:
		ctx.Src = shati
		ctx.Dst = ti
		bicg.resume = 6
		return MatVec, nil
		// Compute As^_i -> t_i.
	case 6:
		bicg.omega = floats.Dot(ctx.Vectors[ti], ctx.Vectors[si]) / floats.Dot(ctx.Vectors[ti], ctx.Vectors[ti])
		floats.AddScaled(ctx.X, bicg.alpha, ctx.Vectors[phati])
		floats.AddScaled(ctx.X, bicg.omega, ctx.Vectors[shati])
		floats.AddScaled(ctx.Vectors[ri], -bicg.omega, ctx.Vectors[ti])
		copy(ctx.Residual, ctx.Vectors[ri])
		ctx.Src = -1
		ctx.Dst = -1
		ctx.Converged = false
		bicg.resume = 7
		return CheckResidual, nil
	case 7:
		if ctx.Converged {
			bicg.resume = 0
			return EndIteration, nil
		}
		if math.Abs(bicg.omega) < dlamchE*dlamchE {
			return NoOperation, errors.New("iterative: omega breakdown")
		}
		bicg.rhoPrev = bicg.rho
		bicg.first = false
		bicg.resume = 1
		return EndIteration, nil

	default:
		panic("iterative: BiCGStab.Init not called")
	}
}
