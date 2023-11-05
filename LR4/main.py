import task as t

print(f"eps: {t.eps_}, tau: {t.tau_}.")
t.solve()
# t.animation()
savepath = f"{t.h_x}_{t.h_y}.png"
t.plot_step("result", "jet", savepath)
# t.plot_exact("jet", "exact_sol.png")
