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

func TestGMRES(t *testing.T) {
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
		market("steam1"),
		market("steam3"),
		market("e05r0000"),
		market("e05r0100"),
		market("e05r0200"),
		market("e05r0300"),
		market("e05r0400"),
		market("e05r0500"),
		market("mcca"),
		market("impcol_a"),
		market("impcol_b"),
		market("impcol_c"),
		market("impcol_d"),
		market("impcol_e"),
		// market("fs_183_1"),
		// market("fs_183_3"),
		market("fs_183_4"),
		market("fs_183_6"),
		// market("mbeacxc"),
		// market("mbeaflw"),
		// market("mbeause"),
		market("west0067"),
		market("west0132"),
		// market("west0156"),
		market("west0167"),
		market("west0381"),
		market("west0479"),
		market("west0497"),
		market("gre__115"),
		market("gre__185"),
		market("gre__343"),
		market("gre_216a"),
		// market("gre_216b"),
		// market("lns__131"),
		// market("lnsp_131"),
		market("hor__131"),
		// market("nnc261"),
		market("arc130"),
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

		// TODO(vladimir-ch): Add tests with non-default Restart. For
		// that we probably need to generate nicer matrices.
		r, err := LinearSolve(A, b, &GMRES{}, Settings{
			MaxIterations: tc.iters,
			Tolerance:     1e-15,
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
