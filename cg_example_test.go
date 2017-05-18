// Copyright ©2016 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iterative_test

import (
	"fmt"
	"math"

	"github.com/vladimir-ch/iterative"
)

func L2Projector(x0, x1 float64, n int, f func(float64) float64) (a iterative.MatrixOps, b []float64) {
	h := (x1 - x0) / float64(n)

	matvec := func(dst, src []float64) {
		h := h
		dst[0] = h / 3 * (src[0] + src[1]/2)
		for i := 1; i < n; i++ {
			dst[i] = h / 3 * (src[i-1]/2 + 2*src[i] + src[i+1]/2)
		}
		dst[n] = h / 3 * (src[n-1]/2 + src[n])
	}

	b = make([]float64, n+1)
	b[0] = f(x0) * h / 2
	for i := 1; i < n; i++ {
		b[i] = f(x0+float64(i)*h) * h
	}
	b[n] = f(x1) * h / 2

	return iterative.MatrixOps{MatVec: matvec}, b
}

func ExampleCG() {
	A, b := L2Projector(0, 1, 10, func(x float64) float64 {
		return x * math.Sin(x)
	})
	res, err := iterative.LinearSolve(A, b, &iterative.CG{}, iterative.Settings{})
	if err != nil {
		fmt.Println("Error:", err)
	} else {
		fmt.Printf("# iterations: %v\n", res.Stats.Iterations)
		fmt.Printf("Final residual: %.6e\n", res.Stats.ResidualNorm)
		fmt.Printf("Solution: %.6f\n", res.X)
	}

	// Output:
	// # iterations: 10
	// Final residual: 6.495861e-08
	// Solution: [-0.003341 0.006678 0.036530 0.085606 0.152981 0.237072 0.337006 0.447616 0.578244 0.682719 0.920847]
}
