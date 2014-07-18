package cg

import "fmt"

func ExampleSolve() {
	a := func(x []float64) []float64 {
		return []float64{3*x[0] - 2*x[1], -2*x[0] + 2*x[1]}
	}
	b := []float64{0, 2}
	x0 := []float64{0, 0}

	x, err := Solve(a, b, x0, 0, 100)
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
