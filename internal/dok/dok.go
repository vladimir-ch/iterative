// Copyright ©2017 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dok

type index struct {
	row, col int
}

type Matrix struct {
	r, c int
	data map[index]float64
}

func New(r, c int) *Matrix {
	return &Matrix{
		r:    r,
		c:    c,
		data: make(map[index]float64),
	}
}

func (m *Matrix) Dims() (r, c int) {
	return m.r, m.c
}

func (m *Matrix) At(i, j int) float64 {
	if i < 0 || m.r <= i {
		panic("row index out of range")
	}
	if j < 0 || m.c <= j {
		panic("column index out of range")
	}
	return m.data[index{i, j}]
}

func (m *Matrix) Set(i, j int, v float64) {
	if i < 0 || m.r <= i {
		panic("row index out of range")
	}
	if j < 0 || m.c <= j {
		panic("column index out of range")
	}
	m.data[index{i, j}] = v
}

func (m *Matrix) MulVec(dst, x []float64) {
	if m.c != len(x) {
		panic("dimension mismatch")
	}
	if m.r != len(dst) {
		panic("dimension mismatch")
	}
	for i := range dst {
		dst[i] = 0
	}
	for ij, aij := range m.data {
		dst[ij.row] += aij * x[ij.col]
	}
}

func (m *Matrix) MulTransVec(dst, x []float64) {
	if m.c != len(dst) {
		panic("dimension mismatch")
	}
	if m.r != len(x) {
		panic("dimension mismatch")
	}
	for i := range dst {
		dst[i] = 0
	}
	for ij, aij := range m.data {
		dst[ij.col] += aij * x[ij.row]
	}
}
