// Copyright ©2017 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package triplet

type triplet struct {
	i, j int
	v    float64
}

type Matrix struct {
	r, c int
	data []triplet
}

func New(r, c int) *Matrix {
	return &Matrix{
		r: r,
		c: c,
	}
}

func (m *Matrix) Dims() (r, c int) {
	return m.r, m.c
}

func (m *Matrix) Append(i, j int, v float64) {
	if i < 0 || m.r <= i {
		panic("row index out of range")
	}
	if j < 0 || m.c <= j {
		panic("column index out of range")
	}
	m.data = append(m.data, triplet{i, j, v})
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
	for _, aij := range m.data {
		dst[aij.i] += aij.v * x[aij.j]
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
	for _, aij := range m.data {
		dst[aij.j] += aij.v * x[aij.i]
	}
}
