package iterative

import (
	"math"
	"math/rand"
	"testing"

	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
	"github.com/gonum/floats"
)

func TestGMRES(t *testing.T) {
	rnd := rand.New(rand.NewSource(1))
	for _, n := range []int{1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500} {
		bi := blas64.Implementation()
		// Generate a general matrix A.
		a := make([]float64, n*n)
		lda := n
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				a[i*lda+j] = rnd.Float64()
			}
		}
		// Compute the right-hand side b so that the vector [1,1,...,1]
		// is the solution.
		x := make([]float64, n)
		for i := range x {
			x[i] = 1
		}
		want := make([]float64, n)
		copy(want, x)
		b := make([]float64, n)
		bi.Dgemv(blas.NoTrans, n, n, 1, a, lda, x, 1, 0, b, 1)
		// Initial estimate is the zero vector.
		for i := range x {
			x[i] = 0
		}
		A := MatrixOps{
			MatVec: func(dst, x []float64) {
				bi.Dgemv(blas.NoTrans, n, n, 1, a, lda, x, 1, 0, dst, 1)
			},
		}
		// TODO(vladimir-ch): Add tests with non-default Restart. For
		// that we probably need to generate nicer matrices.
		r, err := LinearSolve(A, b, &GMRES{}, Settings{})

		if err != nil {
			t.Errorf("Case n=%v: unexpected error %v", n, err)
			continue
		}
		dist := floats.Distance(r.X, want, math.Inf(1))
		if dist > 1e-8 {
			t.Errorf("Case n=%v: unexpected solution, |want-got|=%v", n, dist)
		}
	}
}
