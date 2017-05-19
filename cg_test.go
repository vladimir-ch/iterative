// Copyright ©2017 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iterative

import (
	"math"
	"math/rand"
	"testing"

	"github.com/gonum/floats"
)

func TestCG(t *testing.T) {
	rnd := rand.New(rand.NewSource(1))
	for _, tc := range []testCase{
		randomSPD(1, rnd),
		randomSPD(2, rnd),
		randomSPD(3, rnd),
		randomSPD(4, rnd),
		randomSPD(5, rnd),
		randomSPD(10, rnd),
		randomSPD(20, rnd),
		randomSPD(50, rnd),
		randomSPD(100, rnd),
		randomSPD(200, rnd),
		randomSPD(500, rnd),
		market("nos1"),
		market("nos4"),
		market("nos5"),
		market("bcsstm20"),
		market("bcsstm22"),
	} {
		n := tc.n
		A := tc.a
		// Compute the right-hand side b so that the vector [1,1,...,1]
		// is the solution.
		want := make([]float64, n)
		for i := range want {
			want[i] = 1
		}
		b := make([]float64, n)
		A.MatVec(b, want)
		// Initial estimate is the zero vector.
		x := make([]float64, n)

		r, err := LinearSolve(A, b, &CG{}, Settings{
			MaxIterations: tc.iters,
			Tolerance:     1e-12,
		})
		if err != nil {
			t.Errorf("Case %v (n=%v): unexpected error %v", tc.name, n, err)
		}
		dist := floats.Distance(r.X, want, math.Inf(1))
		if dist > tc.tol {
			t.Errorf("Case %v (n=%v): unexpected solution, |want-got|=%v", tc.name, n, dist)
		}
	}
}
