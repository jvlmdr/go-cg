package pcg

import (
	"fmt"
	"math"

	"github.com/gonum/floats"
	"github.com/jackvalmadre/go-cg/cg"
)

// Minimizes 1/2 x^T A x - b^T x by solving Cinv A x = Cinv b.
// A and Cinv must be symmetric and positive definite.
func Solve(a cg.Func, b []float64, cinv cg.Func, x0 []float64, tol float64, iter int) ([]float64, error) {
	seq := NewSeq(a, b, cinv, x0)
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
	a     cg.Func
	b     []float64
	cinv  cg.Func
	state state
}

type state struct {
	x, r, z, p []float64
}

func (curr state) Next(a, cinv cg.Func) (state, error) {
	var next state
	// Compute t (line search) and take step.
	ap := a(curr.p)
	pap := floats.Dot(curr.p, ap)
	if pap == 0 {
		// Can't divide by zero.
		return state{}, fmt.Errorf("dot(p, A*p) = 0")
	}
	alpha := floats.Dot(curr.r, curr.z) / pap
	next.x = plusScaled(curr.x, alpha, curr.p)
	next.r = plusScaled(curr.r, -alpha, ap)
	next.z = cinv(next.r)
	// Compute beta and take step.
	beta := floats.Dot(next.z, next.r) / floats.Dot(curr.z, curr.r)
	next.p = plusScaled(next.z, beta, curr.p)
	return next, nil
}

func NewSeq(a cg.Func, b []float64, cinv cg.Func, x []float64) *Seq {
	var init state
	// Copy x.
	init.x = clone(x)
	// Initial direction is that of steepest descent.
	ax := a(x)
	init.r = minus(b, ax)
	init.z = cinv(init.r)
	init.p = clone(init.z)
	return &Seq{a, b, cinv, init}
}

func (seq *Seq) Final() bool {
	return sqrnorm(seq.state.r) == 0
}

func (seq *Seq) Iter() error {
	next, err := seq.state.Next(seq.a, seq.cinv)
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
	x := seq.state.x
	return floats.Dot(x, seq.a(x))/2 - floats.Dot(seq.b, x)
}

// Returns ||Ax - b|| / ||b||.
func (seq *Seq) Residual() float64 {
	r := minus(seq.a(seq.state.x), seq.b)
	return math.Sqrt(sqrnorm(r) / sqrnorm(seq.b))
}
