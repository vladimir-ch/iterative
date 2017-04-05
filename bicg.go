// Copyright ©2017 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iterative

import (
	"errors"
	"math"

	"github.com/gonum/floats"
)

// BiCG implements the biconjugate gradient iterative method with
// preconditioning for solving the system of linear equations
//  Ax = b,
// where A is a non-symmetric matrix. For symmetric positive definite systems
// use CG.
//
// BiCG needs MatVec, MatTransVec, PSolve, and PSolveTrans matrix operations.
type BiCG struct {
	first        bool
	rho, rhoPrev float64
	alpha        float64
	resume       int
}

// Init implements the Method interface.
func (bicg *BiCG) Init(dim int) int {
	bicg.first = true
	bicg.resume = 1
	return 6
}

// Iterate implements the Method interface.
func (bicg *BiCG) Iterate(ctx *Context) (Operation, error) {
	const (
		ri  = 0
		rti = 1
		zi  = 2
		zti = 3
		pi  = 4
		pti = 5
		qi  = 2 // z and zt are not needed simultaneously
		qti = 3 // with q and qt.
	)
	switch bicg.resume {
	case 1:
		if bicg.first {
			copy(ctx.Vectors[ri], ctx.Residual)
			copy(ctx.Vectors[rti], ctx.Vectors[ri])
		}
		ctx.Src = ri
		ctx.Dst = zi
		bicg.resume = 2
		return PSolve, nil
		// Solve M z = r_{i-1}
	case 2:
		ctx.Src = rti
		ctx.Dst = zti
		bicg.resume = 3
		return PSolveTrans, nil
		// Solve M^T zt = rt_{i-1}
	case 3:
		bicg.rho = floats.Dot(ctx.Vectors[zi], ctx.Vectors[rti])
		if math.Abs(bicg.rho) < dlamchE*dlamchE {
			bicg.resume = 0
			return NoOperation, errors.New("iterative: rho breakdown")
		}
		if !bicg.first {
			beta := bicg.rho / bicg.rhoPrev
			floats.AddScaled(ctx.Vectors[zi], beta, ctx.Vectors[pi])
			floats.AddScaled(ctx.Vectors[zti], beta, ctx.Vectors[pti])
		}
		copy(ctx.Vectors[pi], ctx.Vectors[zi])
		copy(ctx.Vectors[pti], ctx.Vectors[zti])
		ctx.Src = pi
		ctx.Dst = qi
		bicg.resume = 4
		return MatVec, nil
		// q <- A p
	case 4:
		ctx.Src = pti
		ctx.Dst = qti
		bicg.resume = 5
		return MatTransVec, nil
		// qt <- A^T pt
	case 5:
		bicg.alpha = bicg.rho / floats.Dot(ctx.Vectors[pti], ctx.Vectors[qi])
		floats.AddScaled(ctx.X, bicg.alpha, ctx.Vectors[pi])
		floats.AddScaled(ctx.Residual, -bicg.alpha, ctx.Vectors[qi])
		ctx.Src = -1
		ctx.Dst = -1
		ctx.Converged = false
		bicg.resume = 6
		return CheckResidual, nil
	case 6:
		if ctx.Converged {
			// Make sure calling Iterate again without Init will panic.
			bicg.resume = 0
			return EndIteration, nil
		}
		// Prepare for the next iteration.
		copy(ctx.Vectors[ri], ctx.Residual)
		floats.AddScaled(ctx.Vectors[rti], -bicg.alpha, ctx.Vectors[qti])
		bicg.rhoPrev = bicg.rho
		bicg.first = false
		bicg.resume = 1
		return EndIteration, nil

	default:
		panic("iterative: BiCG.Init not called")
	}
}
