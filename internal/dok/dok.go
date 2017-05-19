// Copyright ©2017 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dok

type DOK struct {
	Rows, Cols int

	data map[index]float64
}

type index struct {
	row, col int
}

func New(r, c int) *DOK {
	return &DOK{
		Rows: r,
		Cols: c,
		data: make(map[index]float64),
	}
}

func (m *DOK) At(i, j int) float64 {
	if i < 0 || m.Rows <= i {
		panic("row index out of range")
	}
	if j < 0 || m.Cols <= j {
		panic("column index out of range")
	}
	return m.data[index{i, j}]
}

func (m *DOK) SetAt(i, j int, v float64) {
	if i < 0 || m.Rows <= i {
		panic("row index out of range")
	}
	if j < 0 || m.Cols <= j {
		panic("column index out of range")
	}
	m.data[index{i, j}] = v
}

func (m *DOK) MulVec(dst, x []float64) {
	if m.Cols != len(x) {
		panic("dimension mismatch")
	}
	if m.Rows != len(dst) {
		panic("dimension mismatch")
	}
	for ij, aij := range m.data {
		dst[ij.row] += aij * x[ij.col]
	}
}

func (m *DOK) MulTransVec(dst, x []float64) {
	if m.Cols != len(dst) {
		panic("dimension mismatch")
	}
	if m.Rows != len(x) {
		panic("dimension mismatch")
	}
	for ij, aij := range m.data {
		dst[ij.col] += aij * x[ij.row]
	}
}
