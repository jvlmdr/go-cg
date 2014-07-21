package cg

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/gonum/blas/goblas"
	"github.com/gonum/matrix/mat64"
)

func init() {
	mat64.Register(goblas.Blas{})
}

func ExampleSolve() {
	a := func(x []float64) []float64 {
		return []float64{3*x[0] - 2*x[1], -2*x[0] + 2*x[1]}
	}
	b := []float64{0, 2}

	x0 := []float64{0, 0}
	x, err := Solve(a, b, x0, 0, 2)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(x)
	// Output:
	// [2 3]
}

func ExampleSeq() {
	const iter = 2
	a := func(x []float64) []float64 {
		return []float64{3*x[0] - 2*x[1], -2*x[0] + 2*x[1]}
	}
	b := []float64{0, 2}

	x0 := []float64{0, 0}
	seq := NewSeq(a, b, x0)
	for i := 0; i < iter && !seq.Final(); i++ {
		if err := seq.Iter(); err != nil {
			fmt.Println("error:", err)
			return
		}
	}
	fmt.Println(seq.Solution())
	// Output:
	// [2 3]
}

func TestSolve(t *testing.T) {
	const n = 1000
	want := randVec(n)
	a := randMat(n)
	// f(x) = A x
	f := func(x []float64) []float64 {
		xvec := mat64.Vec(x)
		y := mat64.Vec(make([]float64, n))
		y.Mul(a, &xvec)
		return y
	}
	b := f(want)
	x0 := make([]float64, n)

	got, err := Solve(f, b, x0, 0, n/10)
	if err != nil {
		t.Fatal("error:", err)
	}
	checkEqual(t, want, got, 1e-6)
}

func randVec(n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = rand.NormFloat64()
	}
	return x
}

func randMat(n int) mat64.Matrix {
	m := 2 * n
	v := mat64.NewDense(m, n, nil)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			v.Set(i, j, rand.NormFloat64())
		}
	}
	// A <- V' V
	a := mat64.NewDense(n, n, nil)
	vt := mat64.NewDense(n, m, nil)
	vt.TCopy(v)
	a.Mul(vt, v)
	return a
}

func checkEqual(t *testing.T, want, got []float64, eps float64) {
	if len(want) != len(got) {
		t.Fatalf("lengths differ: want %d, got %d", len(want), len(got))
	}
	for i := range want {
		if math.Abs(want[i]-got[i]) > eps {
			t.Errorf("different: at %d: want %g, got %g", i, want[i], got[i])
		}
	}
}
