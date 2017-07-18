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

func TestBiCGSTAB(t *testing.T) {
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
		market("nos1", 1e-9),
		market("nos4", 1e-12),
		market("nos5", 1e-12),
		market("bcsstm20", 1e-9),
		market("bcsstm22", 1e-10),
		// market("steam1", 1e-8),
		// market("steam3", 1e-8),
		market("e05r0000", 1e-10),
		market("e05r0100", 1e-10),
		// market("e05r0200", 1e-10),
		// market("e05r0300", 1e-10),
		// market("e05r0400", 1e-10),
		// market("e05r0500", 1e-10),
		// market("mcca", 1e-5),
		// market("impcol_a", 1e-8),
		// market("impcol_b", 1e-9),
		// market("impcol_c", 1e-10),
		// market("impcol_d", 1e-12),
		// market("impcol_e", 1e-11),
		// market("fs_183_1", 1e-12),
		// market("fs_183_3", 1e-12),
		// market("fs_183_4", 1e-5),
		// market("fs_183_6", 1e-4),
		// market("mbeacxc", 1e-12),
		// market("mbeaflw", 1e-12),
		// market("mbeause", 1e-12),
		// market("west0067", 1e-11),
		// market("west0132", 1e-6),
		// market("west0156", 1e-12),
		// market("west0167", 1e-8),
		// market("west0381", 1e-11),
		// market("west0479", 1e-6),
		// market("west0497", 1e-7),
		market("gre__115", 1e-12),
		market("gre__185", 1e-9),
		// market("gre__343", 1e-12),
		// market("gre_216a", 1e-12),
		// market("gre_216b", 1e-12),
		// market("lns__131", 1e-12),
		// market("lnsp_131", 1e-12),
		// market("hor__131", 1e-12),
		// market("nnc261", 1e-12),
		market("arc130", 1e-4),
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

		r, err := LinearSolve(A, b, &BiCGSTAB{}, Settings{
			MaxIterations: 10 * tc.iters,
			Tolerance:     1e-14,
		})
		if err != nil {
			t.Errorf("Case %v (n=%v): unexpected error %v", tc.name, n, err)
			continue
		}
		dist := floats.Distance(r.X, want, math.Inf(1))
		if dist > tc.tol {
			t.Errorf("Case %v (n=%v): unexpected solution, |want-got|=%v", tc.name, n, dist)
		}
	}
}
