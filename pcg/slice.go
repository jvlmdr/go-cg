package pcg

import "github.com/gonum/floats"

func clone(x []float64) []float64 {
	y := make([]float64, len(x))
	copy(y, x)
	return y
}

func dot(x, y []float64) float64 {
	if len(x) == 1 {
		return x[0] * y[0]
	}
	n := len(x)
	m := (n + 1) / 2
	return dot(x[0:m], y[0:m]) + dot(x[m:n], y[m:n])
}

func sqrnorm(x []float64) float64 {
	return dot(x, x)
}

func plusScaled(a []float64, k float64, b []float64) []float64 {
	dst := make([]float64, len(a))
	floats.AddScaledTo(dst, a, k, b)
	return dst
}

func minus(a, b []float64) []float64 {
	dst := make([]float64, len(a))
	floats.SubTo(dst, a, b)
	return dst
}
