package pcg

import "fmt"

func ExampleSolve() {
	a := func(x []float64) []float64 {
		return []float64{3*x[0] - 2*x[1], -2*x[0] + 2*x[1]}
	}
	b := []float64{0, 2}
	cinv := func(x []float64) []float64 { return x }

	x0 := []float64{0, 0}
	x, err := Solve(a, b, cinv, x0, 0, 2, nil)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(x)
	// Output:
	// [2 3]
}
