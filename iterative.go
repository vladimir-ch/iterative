// Copyright ©2016 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iterative

import (
	"errors"
	"time"

	"github.com/gonum/floats"
)

type Matrix struct {
	MatVec      func(dst, src []float64)
	MatTransVec func(dst, src []float64)
}

type Settings struct {
	Tolerance   float64
	Iterations  int
	X0          []float64
	PSolve      func(dst, rhs []float64) error
	PSolveTrans func(dst, rhs []float64) error
}

type Operation uint64

const (
	NoOperation Operation = 0
	MatVec      Operation = 1 << (iota - 1)
	MatTransVec
	PSolve
	PSolveTrans
	ComputeResidual
	CheckConvergence
	Iteration
)

type Method interface {
	Init(dim int) (lwork int)
	Iterate(*Context) (Operation, error)
}

type Context struct {
	X         []float64
	Residual  []float64
	Converged bool
	Work      [][]float64
	Src, Dst  int
}

type Stats struct {
	Iterations int
	MatVec     int
	PSolve     int
	Residual   float64
	StartTime  time.Time
	Runtime    time.Duration
}

type Result struct {
	X     []float64
	Stats Stats
}

func Solve(a Matrix, b []float64, method Method, settings Settings) (Result, error) {
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

	ctx := &Context{
		X:        make([]float64, dim),
		Residual: make([]float64, dim),
	}
	if settings.X0 != nil {
		copy(ctx.X, settings.X0)
		a.MatVec(ctx.Residual, ctx.X)
		floats.AddScaledTo(ctx.Residual, b, -1, ctx.Residual) // r = b - Ax
	} else {
		copy(ctx.Residual, b) // r = b
	}

	var err error
	if floats.Norm(ctx.Residual, 2) >= settings.Tolerance {
		err = iterate(a, b, ctx, settings, method, &stats)
	}

	stats.Runtime = time.Since(stats.StartTime)
	return Result{
		X:     ctx.X,
		Stats: stats,
	}, err
}

func iterate(a Matrix, b []float64, ctx *Context, settings Settings, method Method, stats *Stats) error {
	dim := len(ctx.X)
	bnorm := floats.Norm(b, 2)
	if bnorm == 0 {
		bnorm = 1
	}

	lwork := method.Init(dim)
	ctx.Work = make([][]float64, lwork)
	for i := range ctx.Work {
		ctx.Work[i] = make([]float64, dim)
	}

	for {
		op, err := method.Iterate(ctx)
		if err != nil {
			return err
		}

		switch op {
		case NoOperation:

		case ComputeResidual:
			a.MatVec(ctx.Residual, ctx.X)
			floats.AddScaledTo(ctx.Residual, b, -1, ctx.Residual)

		case MatVec, MatTransVec:
			dst := ctx.Work[ctx.Dst]
			src := ctx.Work[ctx.Src]
			if op == MatVec {
				a.MatVec(dst, src)
			} else {
				a.MatTransVec(dst, src)
			}
			stats.MatVec++

		case PSolve, PSolveTrans:
			dst := ctx.Work[ctx.Dst]
			src := ctx.Work[ctx.Src]
			if settings.PSolve == nil {
				copy(dst, src)
				continue
			}
			if op == PSolve {
				err = settings.PSolve(dst, src)
			} else {
				err = settings.PSolveTrans(dst, src)
			}
			if err != nil {
				return err
			}
			stats.PSolve++

		case CheckConvergence:
			stats.Residual = floats.Norm(ctx.Residual, 2) / bnorm
			ctx.Converged = stats.Residual < settings.Tolerance

		case Iteration:
			stats.Iterations++
			if ctx.Converged {
				return nil
			}
			if stats.Iterations == settings.Iterations {
				return errors.New("iterative: iteration limit reached")
			}

		default:
			panic("iterate: invalid operation")
		}
	}
}

func DefaultSettings() Settings {
	return Settings{
		Tolerance: 1e-6,
	}
}

func defaultSettings(s *Settings, dim int) {
	if s.Tolerance == 0 {
		s.Tolerance = 1e-6
	}
	if s.Iterations == 0 {
		s.Iterations = 2 * dim
	}
}
