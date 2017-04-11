// Copyright ©2016 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package iterative provides iterative algorithms for solving linear systems.
package iterative

import (
	"errors"
	"time"

	"github.com/gonum/floats"
)

// MatrixOps describes the matrix of the
// linear system.
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
	Tolerance float64

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

// Operation specifies the type of operation.
type Operation uint64

// Operations commanded by Method.Iterate.
const (
	NoOperation Operation = 0

	// Multiply A*x where x is stored
	// in Context.Src and the result will
	// be stored in Context.Dst.
	MatVec Operation = 1 << (iota - 1)

	// Multiply A^T*x where x is stored
	// in Context.Src and the result will
	// be stored in Context.Dst.
	MatTransVec

	// Do the preconditioner solve
	//  M z = r,
	// where r is stored in Context.Src,
	// and store the solution z in
	// Context.Dst.
	PSolve

	// Do the preconditioner solve
	//  M^T z = r,
	// where r is stored in Context.Src,
	// and store the solution z in
	// Context.Dst.
	PSolveTrans

	// Compute b - A*x where x is stored
	// in Context.X and store the result
	// into Context.Dst.
	ComputeResidual

	// Check convergence using the
	// current approximation in Context.X
	// and the residual in Context.ResidualNorm.
	// If convergence is detected,
	// Context.Converged must be set to
	// true before calling Method.Iterate
	// again.
	CheckResidualNorm

	// EndIteration indicates that Method
	// has finished what it considers to
	// be one iteration. It can be used
	// to update an iteration counter. If
	// Context.Converged is true, the
	// iterative process must be
	// terminated, and Method.Init must
	// be called before calling
	// Method.Iterate again.
	EndIteration
)

// Method is an iterative method that produces a sequence x_i of vectors of
// dimension dim converging to the vector x satisfying an dim×dim system of
// linear equations
//  A x = b.
type Method interface {
	Init(dim int)
	Iterate(*Context) (Operation, error)
}

type Context struct {
	X            []float64
	Residual     []float64
	ResidualNorm float64
	Converged    bool

	Src, Dst []float64
}

type Stats struct {
	Iterations   int
	MatVec       int
	PSolve       int
	ResidualNorm float64
	StartTime    time.Time
	Runtime      time.Duration
}

type Result struct {
	X     []float64
	Stats Stats
}

func Solve(a MatrixOps, b []float64, method Method, settings Settings) (Result, error) {
	stats := Stats{StartTime: time.Now()}

	dim := len(b)
	switch {
	case dim == 0:
		panic("iterative: zero dimension")
	case a.MatVec == nil:
		panic("iterative: nil matrix-vector multiplication")
	case settings.X0 != nil && len(settings.X0) != dim:
		panic("iterative: mismatched length of initial guess")
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
			floats.AddScaledTo(ctx.Residual, b, -1, ctx.Residual) // r = b - Ax

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

func DefaultSettings() Settings {
	return Settings{
		Tolerance: 1e-8,
	}
}

func defaultSettings(s *Settings, dim int) {
	if s.Tolerance == 0 {
		s.Tolerance = 1e-8
	}
	if s.MaxIterations == 0 {
		s.MaxIterations = 2 * dim
	}
}

func reuse(v []float64, n int) []float64 {
	if cap(v) < n {
		return make([]float64, n)
	}
	return v[:n]
}

const dlamchE = 1.0 / (1 << 53)
