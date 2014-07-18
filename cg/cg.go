package cg

import (
	"fmt"
	"math"

	"github.com/gonum/floats"
)

// Symmetric linear map from R^n to R^n.
type Func func(x []float64) []float64

func Solve(a Func, b, x []float64, tol float64, iter int) ([]float64, error) {
	seq := NewSeq(a, b, x)
	for i := 0; i < iter && !seq.Final(); i++ {
		if err := seq.Iter(); err != nil {
			return nil, err
		}
		if seq.Residual() <= tol {
			break
		}
	}
	return seq.Solution(), nil
}

type Seq struct {
	a     Func
	b     []float64
	state state
}

type state struct {
	x, r, p []float64
}

func (curr state) Next(a Func) (state, error) {
	var next state
	// Compute alpha (line search) and take step.
	ap := a(curr.p)
	pap := floats.Dot(curr.p, ap)
	if pap == 0 {
		// Can't divide by zero.
		return state{}, fmt.Errorf("dot(p, A*p) = 0")
	}
	alpha := sqrnorm(curr.r) / pap
	next.x = plusScaled(curr.x, alpha, curr.p)
	next.r = plusScaled(curr.r, -alpha, ap)
	// Compute beta and take step.
	beta := sqrnorm(next.r) / sqrnorm(curr.r)
	next.p = plusScaled(next.r, beta, curr.p)
	return next, nil
}

func NewSeq(a Func, b, x []float64) *Seq {
	var init state
	init.x = clone(x)
	// Initial direction is that of steepest descent.
	ax := a(init.x)
	init.r = minus(b, ax)
	init.p = clone(init.r)
	return &Seq{a, b, init}
}

func (seq *Seq) Final() bool {
	r := seq.state.r
	return sqrnorm(r) == 0
}

// Performs one iteration.
func (seq *Seq) Iter() error {
	next, err := seq.state.Next(seq.a)
	if err != nil {
		return err
	}
	seq.state = next
	return nil
}

func (seq *Seq) Solution() []float64 {
	return seq.state.x
}

// Returns 1/2 x' A x - b' x.
func (seq *Seq) Objective() float64 {
	// Print summary.
	x := seq.state.x
	return floats.Dot(x, seq.a(x))/2 - floats.Dot(seq.b, x)
}

// Returns ||Ax - b|| / ||b||.
func (seq *Seq) Residual() float64 {
	// Print summary.
	r := minus(seq.a(seq.state.x), seq.b)
	return math.Sqrt(sqrnorm(r) / sqrnorm(seq.b))
}
