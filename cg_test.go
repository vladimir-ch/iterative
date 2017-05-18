package iterative

import (
	"math"
	"math/rand"
	"testing"

	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
	"github.com/gonum/floats"
)

func TestCG(t *testing.T) {
	rnd := rand.New(rand.NewSource(1))
	for _, n := range []int{1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500} {
		// Generate a symmetric positive-definite matrix A.
		a := make([]float64, n*n)
		lda := n
		for i := 0; i < n; i++ {
			for j := i; j < n; j++ {
				a[i*lda+j] = rnd.Float64()
			}
		}
		for i := 0; i < n; i++ {
			a[i*lda+i] += float64(n)
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
		bi := blas64.Implementation()
		bi.Dsymv(blas.Upper, n, 1, a, lda, x, 1, 0, b, 1)
		// Initial estimate is the zero vector.
		for i := range x {
			x[i] = 0
		}

		A := MatrixOps{
			MatVec: func(dst, x []float64) {
				bi.Dsymv(blas.Upper, n, 1, a, lda, x, 1, 0, dst, 1)
			},
		}
		r, err := LinearSolve(A, b, &CG{}, Settings{Tolerance: 1e-14})

		if err != nil {
			t.Errorf("Case n=%v: unexpected error %v", n, err)
		}
		dist := floats.Distance(r.X, want, math.Inf(1))
		if dist > 1e-10 {
			t.Errorf("Case n=%v: unexpected solution, |want-got|=%v", n, dist)
		}
	}
}
