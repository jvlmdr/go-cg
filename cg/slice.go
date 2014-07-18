package cg

import "github.com/gonum/floats"

func clone(x []float64) []float64 {
	y := make([]float64, len(x))
	copy(y, x)
	return y
}

func sqrnorm(x []float64) float64 {
	return floats.Dot(x, x)
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
