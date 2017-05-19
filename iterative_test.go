// Copyright ©2017 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iterative

import (
	"compress/gzip"
	"math/rand"
	"os"

	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
	"github.com/vladimir-ch/iterative/internal/mmarket"
)

type testCase struct {
	name  string
	n     int
	iters int
	tol   float64
	a     MatrixOps
}

// randomSPD returns a random symmetric positive-definite matrix of order n.
func randomSPD(n int, rnd *rand.Rand) testCase {
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
	return testCase{
		name:  "randomSPD",
		n:     n,
		iters: 2 * n,
		tol:   1e-10,
		a: MatrixOps{
			MatVec: func(dst, x []float64) {
				bi := blas64.Implementation()
				bi.Dsymv(blas.Upper, n, 1, a, lda, x, 1, 0, dst, 1)
			},
			MatTransVec: func(dst, x []float64) {
				bi := blas64.Implementation()
				bi.Dsymv(blas.Upper, n, 1, a, lda, x, 1, 0, dst, 1)
			},
		},
	}
}

// market returns a test matrix from the Matrix Market.
func market(name string) testCase {
	f, err := os.Open("testdata/" + name + ".mtx.gz")
	if err != nil {
		panic(err)
	}
	gz, err := gzip.NewReader(f)
	if err != nil {
		panic(err)
	}
	m, err := mmarket.NewReader(gz).Read()
	if err != nil {
		panic(err)
	}
	n, _ := m.Dims()
	return testCase{
		name:  name,
		n:     n,
		iters: 10 * n,
		tol:   1e-7,
		a: MatrixOps{
			MatVec:      m.MulVec,
			MatTransVec: m.MulTransVec,
		},
	}
}
