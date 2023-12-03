import task as t
import plot as p

res = t.solve()
p.plot(res, name="res")#, plot_exact=True)
p.plot_error(res, name="error")
p.plot_f(name="f")
