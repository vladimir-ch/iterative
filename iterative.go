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

// Method is an iterative method that produces a sequence of vectors converging
// to the vector x satisfying a system of linear equations
//  A x = b,
// where A is non-singular dim×dim matrix, and x and b are vectors of dimension
// dim.
//
// Method uses a reverse-communication interface between the iterative algorithm
// and the caller. Method acts as a client that commands the caller to perform
// needed operations via Operation returned from Iterate methods. This provides
// independence of Method on representation of the matrix A, and enables
// automation of common operations like checking for convergence and maintaining
// statistics.
type Method interface {
	// Init initializes the method for solving an dim×dim linear system.
	Init(dim int)

	// Iterate retrieves data from Context, updates it, and returns the next
	// operation. The caller must perform the Operation using data in
	// Context, and depending on the state call Iterate again.
	Iterate(*Context) (Operation, error)
}

// Context mediates the communication between a Method and the caller. It must
// not be modified or accessed apart from the commanded Operations.
type Context struct {
	// X is the current approximate solution. On the first call to
	// Method.Iterate, X must contain the initial estimate. Method must
	// update X with the current estimate when it commands ComputeResidual
	// and EndIteration.
	X []float64
	// Residual is the current residual b-A*x. On the first call to
	// Method.Iterate, Residual must contain the initial residual.
	// TODO(vladimir-ch): Consider whether the behavior should also include:
	// Method must update Residual with the current value of b-A*x when it
	// commands EndIteration.
	Residual []float64
	// ResidualNorm is (an estimate of) the norm of the current residual.
	// Method must update it when it commands CheckResidualNorm. It does
	// not have to be equal to the norm of Residual, some methods (e.g.,
	// GMRES) can estimate the residual norm without forming the residual
	// itself.
	ResidualNorm float64
	// Converged indicates to Method that the ResidualNorm satisfies the
	// stopping criterion as a result of CheckResidualNorm operation.
	// If a Method commands EndIteration with Converged true, the caller
	// must not call Method.Iterate again without calling Method.Init first.
	Converged bool

	// Src and Dst are the source and destination vectors for various
	// Operations.
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
