// Copyright ©2017 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iterative

import (
	"errors"
	"time"

	"github.com/gonum/floats"
)

// MatrixOps describes the matrix of the
// linear system in terms of A*x and A^T*x
// operations.
type MatrixOps struct {
	// Compute A*x and store the result
	// into dst.
	// It must be non-nil.
	MatVec func(dst, x []float64)

	// Compute A^T*x and store the result
	// into dst.
	// If the matrix is symmetric and a
	// solver for symmetric systems is
	// used (like CG), MatTransVec can be
	// nil.
	MatTransVec func(dst, x []float64)
}

// Settings holds various settings for
// solving a linear system.
type Settings struct {
	// X0 is an initial guess.
	// If it is nil, the zero vector will
	// be used.
	// If it is not nil, the length of X0
	// must be equal to the dimension of
	// the system.
	X0 []float64

	// Tolerance specifies error
	// tolerance for the final
	// approximate solution produced by
	// the iterative method.
	// Tolerance must be smaller than one
	// and greater than the machine
	// epsilon.
	//
	// If NormA is not zero, the stopping
	// criterion used will be
	//  |r_i| < Tolerance * (|A|*|x_i| + |b|),
	// If NormA is zero (not available),
	// the stopping criterion will be
	//  |r_i| < Tolerance * |b|.
	Tolerance float64

	// NormA is an estimate of a norm |A|
	// of A, for example, an approximation
	// of the largest entry. Zero value
	// means that the norm is unknown,
	// and it will not be used in the
	// stopping criterion.
	NormA float64

	// MaxIterations is the limit on the
	// number of iterations.
	// If it is zero, it will be set to
	// twice the dimension of the system.
	MaxIterations int

	// PSolve describes the
	// preconditioner solve that stores
	// into dst the solution of the
	// system
	//  M z = rhs.
	// If it is nil, no preconditioning
	// will be used (M is the
	// identitify).
	PSolve func(dst, rhs []float64) error

	// PSolveTrans describes the
	// preconditioner solve that stores
	// into dst the solution of the
	// system
	//  M^T z = rhs.
	// If it is nil, no preconditioning
	// will be used (M is the
	// identitify).
	PSolveTrans func(dst, rhs []float64) error
}

func defaultSettings(s *Settings, dim int) {
	if s.Tolerance == 0 {
		s.Tolerance = 1e-6
	}
	if s.MaxIterations == 0 {
		s.MaxIterations = 2 * dim
	}
}

// Result holds the result of an iterative solve.
type Result struct {
	// X is the approximate solution.
	X []float64
	// Stats holds the statistics of the
	// solve.
	Stats Stats
}

// Stats holds statistics about an iterative solve.
type Stats struct {
	// Iterations is the number of
	// iteration done by Method.
	Iterations int
	// MatVec is the number of MatVec and
	// MatTransVec operations commanded
	// by a Method.
	MatVec int
	// PSolve is the number of PSolve and
	// PSolveTrans operations commanded
	// by a Method.
	PSolve int
	// ResidualNorm is the final norm of
	// the residual.
	ResidualNorm float64
	// StartTime is an approximate time
	// when the solve was started.
	StartTime time.Time
	// Runtime is an approximate duration
	// of the solve.
	Runtime time.Duration
}

// LinearSolve solves the system of n linear equations
//  A*x = b,
// where the n×n matrix A is represented by the matrix-vector operations in a.
// The dimension of the problem n is determined by the length of b.
//
// method is an iterative method used for finding an approximate solution of the
// linear system. It must not be nil. The operations in a must provide what the
// method needs.
//
// settings provide means for adjusting the iterative process. Zero values of
// the fields mean default values.
func LinearSolve(a MatrixOps, b []float64, method Method, settings Settings) (Result, error) {
	stats := Stats{StartTime: time.Now()}

	dim := len(b)
	if a.MatVec == nil {
		panic("iterative: nil matrix-vector multiplication")
	}
	if settings.X0 != nil && len(settings.X0) != dim {
		panic("iterative: mismatched length of initial guess")
	}

	if dim == 0 {
		return Result{Stats: stats}, nil
	}

	defaultSettings(&settings, dim)
	if settings.Tolerance < dlamchE || 1 <= settings.Tolerance {
		panic("iterative: invalid tolerance")
	}

	ctx := &Context{
		X:        make([]float64, dim),
		Residual: make([]float64, dim),
	}
	if settings.X0 != nil {
		copy(ctx.X, settings.X0)
		a.MatVec(ctx.Residual, ctx.X)
		stats.MatVec++
		floats.AddScaledTo(ctx.Residual, b, -1, ctx.Residual) // r = b - Ax
	} else {
		copy(ctx.Residual, b) // r = b
	}

	ctx.ResidualNorm = floats.Norm(ctx.Residual, 2)
	var err error
	if ctx.ResidualNorm >= settings.Tolerance {
		err = iterate(a, b, ctx, settings, method, &stats)
	}

	stats.Runtime = time.Since(stats.StartTime)
	return Result{
		X:     ctx.X,
		Stats: stats,
	}, err
}

func iterate(a MatrixOps, b []float64, ctx *Context, settings Settings, method Method, stats *Stats) error {
	dim := len(ctx.X)
	bnorm := floats.Norm(b, 2)
	if bnorm == 0 {
		bnorm = 1
	}

	method.Init(dim)

	for {
		op, err := method.Iterate(ctx)
		if err != nil {
			return err
		}

		switch op {
		case NoOperation:

		case ComputeResidual:
			a.MatVec(ctx.Residual, ctx.X)
			stats.MatVec++
			floats.AddScaledTo(ctx.Residual, b, -1, ctx.Residual)

		case MatVec, MatTransVec:
			if op == MatVec {
				a.MatVec(ctx.Dst, ctx.Src)
			} else {
				a.MatTransVec(ctx.Dst, ctx.Src)
			}
			stats.MatVec++

		case PSolve, PSolveTrans:
			if settings.PSolve == nil {
				copy(ctx.Dst, ctx.Src)
				continue
			}
			if op == PSolve {
				err = settings.PSolve(ctx.Dst, ctx.Src)
			} else {
				err = settings.PSolveTrans(ctx.Dst, ctx.Src)
			}
			if err != nil {
				return err
			}
			stats.PSolve++

		case CheckResidualNorm:
			ctx.Converged = ctx.ResidualNorm/bnorm < settings.Tolerance

		case EndIteration:
			stats.Iterations++
			stats.ResidualNorm = ctx.ResidualNorm
			if ctx.Converged {
				return nil
			}
			if stats.Iterations == settings.MaxIterations {
				return errors.New("iterative: iteration limit reached")
			}

		default:
			panic("iterate: invalid operation")
		}
	}
}
