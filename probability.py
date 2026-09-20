import manim as m
import numpy as np
from math import sin, ceil, gamma, sqrt, pi, exp
from scipy import stats
from scipy.special import erf
from scipy.stats import norm
from theming import LinearTransformationScene_, ThreeDScene_, Scene_


# ============================================================
# Вспомогательные плотности
# ============================================================
def _norm_pdf(x, mu=0.0, sigma=1.0):
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * sqrt(2 * pi))


def _beta_pdf(x, a, b):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    mask = (x > 0) & (x < 1)
    out[mask] = (x[mask] ** (a - 1)) * ((1 - x[mask]) ** (b - 1)) / (
        gamma(a) * gamma(b) / gamma(a + b)
    )
    return out


def _t_pdf(x, nu):
    x = np.asarray(x, dtype=float)
    coef = gamma((nu + 1) / 2) / (sqrt(nu * pi) * gamma(nu / 2))
    return coef * (1 + x ** 2 / nu) ** (-(nu + 1) / 2)


def _pareto_pdf(x, xm=1.0, alpha=1.5):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    mask = x >= xm
    out[mask] = alpha * xm ** alpha / x[mask] ** (alpha + 1)
    return out


# ============================================================
# 1. Теорема Байеса: 10000 человечков
# ============================================================
class BayesTheorem(Scene_):
    """
    Квадрат из 10000 точек: 100 больных (красных), из них 99 с «+».
    Из 9900 здоровых 99 ложно-положительных. Итог: 198 «+», половина больна.
    """
    def construct(self):
        np.random.seed(42)
        n = 100
        grid_side = 2.6
        spacing = grid_side / n

        sick_indices = set(np.random.choice(n * n, 100, replace=False))

        dots = m.VGroup()
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                is_sick = idx in sick_indices
                color = m.RED if is_sick else m.BLUE_E
                dot = m.Dot(
                    [j * spacing - grid_side / 2,
                     -i * spacing + grid_side / 2,
                     0],
                    color=color,
                    radius=0.0055,
                    fill_opacity=1.0
                )
                dots.add(dot)

        dots.shift(m.LEFT * 3.6)
        self.add(dots)

        title = m.Text("10000 человек: 100 больны (красные)",
                       font_size=22, color=m.WHITE).to_edge(m.UP)
        self.add(title)

        # Правая панель с числами
        line1 = m.Text("Больны (100): 99 +, 1 -",
                          font_size=22, color=m.RED)
        line2 = m.Text("Здоровы (9900): 99 +, 9801 -",
                          font_size=22, color=m.BLUE)
        line3 = m.Text("Всего '+' : 99 + 99 = 198",
                          font_size=22, color=m.YELLOW)
        line4 = m.Text(r"P(болен | +) = 99 / 198 = 0.5",
                          font_size=30, color=m.GREEN)

        panel = m.VGroup(line1, line2, line3, line4)
        panel.arrange(m.DOWN, buff=0.55, aligned_edge=m.LEFT)
        panel.shift(m.RIGHT * 3.4 + m.UP * 0.3)
        self.add(panel)

        caption = m.Text("Половина «положительных» — реально больны",
                         font_size=22, color=m.WHITE).to_edge(m.DOWN)
        self.add(caption)
        self.wait(2)


# ============================================================
# 2. Интеграл нормальной плотности на [a, b]
# ============================================================
class NormalIntegral(Scene_):
    """
    Площадь под нормальной кривой на отрезке [a, b] = вероятность.
    """
    def construct(self):
        axes = m.Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 0.45, 0.1],
            x_length=9,
            y_length=5.5,
            axis_config={"color": m.GRAY}
        )
        axes.add_coordinates()
        self.add(axes)

        pdf = lambda x: _norm_pdf(x, 0, 1)
        curve = axes.plot(pdf, x_range=[-4, 4], color=m.BLUE)
        label = m.MathTex(r"p(x)=\frac{1}{\sqrt{2\pi}}e^{-x^2/2}",
                          color=m.BLUE, font_size=28).to_corner(m.UL)
        self.add(curve, label)

        a = m.ValueTracker(-1.0)
        b = m.ValueTracker(1.0)

        def build_area(a_val, b_val):
            xs = np.linspace(a_val, b_val, 80)
            pts = [axes.coords_to_point(x, 0.0) for x in xs]
            pts += [axes.coords_to_point(x, float(pdf(x))) for x in reversed(xs)]
            poly = m.Polygon(*pts, color=m.YELLOW, fill_opacity=0.5,
                             stroke_width=0)
            return poly

        area = build_area(a.get_value(), b.get_value())
        self.add(area)

        a_line = m.DashedLine(
            axes.coords_to_point(a.get_value(), 0),
            axes.coords_to_point(a.get_value(), float(pdf(a.get_value()))),
            color=m.RED
        )
        b_line = m.DashedLine(
            axes.coords_to_point(b.get_value(), 0),
            axes.coords_to_point(b.get_value(), float(pdf(b.get_value()))),
            color=m.RED
        )
        self.add(a_line, b_line)

        integral_label = m.MathTex(
            r"\int_{a}^{b} p(x)\,dx",
            font_size=30, color=m.YELLOW
        ).to_corner(m.UR)
        self.add(integral_label)

        # Числовое значение
        val_label = m.DecimalNumber(0.683, num_decimal_places=3,
                                    color=m.YELLOW, font_size=36)
        val_label.next_to(integral_label, m.DOWN, buff=0.4)
        self.add(val_label)

        a_label = m.MathTex("a", color=m.RED).next_to(
            axes.coords_to_point(a.get_value(), 0), m.DOWN)
        b_label = m.MathTex("b", color=m.RED).next_to(
            axes.coords_to_point(b.get_value(), 0), m.DOWN)
        self.add(a_label, b_label)

        def update_all(_):
            area.become(build_area(a.get_value(), b.get_value()))
            a_line.put_start_and_end_on(
                axes.coords_to_point(a.get_value(), 0),
                axes.coords_to_point(a.get_value(), float(pdf(a.get_value())))
            )
            b_line.put_start_and_end_on(
                axes.coords_to_point(b.get_value(), 0),
                axes.coords_to_point(b.get_value(), float(pdf(b.get_value())))
            )
            a_label.move_to(axes.coords_to_point(a.get_value(), 0) + m.DOWN * 0.3)
            b_label.move_to(axes.coords_to_point(b.get_value(), 0) + m.DOWN * 0.3)

        area.add_updater(update_all)

        self.play(
            a.animate.set_value(-2.0),
            b.animate.set_value(0.5),
            run_time=3
        )
        self.play(
            a.animate.set_value(-0.5),
            b.animate.set_value(2.5),
            run_time=3
        )

        caption = m.Text("Площадь под кривой = вероятность",
                         font_size=22, color=m.WHITE).to_edge(m.DOWN)
        self.add(caption)
        self.wait(2)


# ============================================================
# 3. Совместное, маргинальные и условное распределения
# ============================================================
class JointMarginalConditional(Scene_):
    """
    Тепловая карта p(x,y), маргинальные p(x), p(y) и условное p(y|x0).
    """
    def construct(self):
        # Плотность совместного распределения (двумерный гауссиан со сдвигом)
        mu1 = np.array([0.3, -0.2])
        Sigma = np.array([[0.5, 0.35], [0.35, 0.5]])
        Sinv = np.linalg.inv(Sigma)

        def joint(x, y):
            v = np.array([x - mu1[0], y - mu1[1]])
            return float(np.exp(-0.5 * v @ Sinv @ v))

        # Сетка тепловой карты
        N = 30
        side = 3.2
        xmin, xmax = -1.8, 1.8
        ymin, ymax = -1.8, 1.8

        center = m.LEFT * 1.6 + m.DOWN * 0.3

        cells = m.VGroup()
        vals = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                xc = xmin + (j + 0.5) * (xmax - xmin) / N
                yc = ymin + (i + 0.5) * (ymax - ymin) / N
                val = joint(xc, yc)
                vals[i, j] = val
                # Инвертируем по y для правильной ориентации
                cell = m.Rectangle(
                    width=side / N,
                    height=side / N,
                    fill_opacity=1.0,
                    stroke_width=0
                )
                cell.set_fill(m.interpolate_color(m.BLACK, m.YELLOW, min(1, val * 2.5)))
                cell.move_to(center + m.RIGHT * (
                    xc / (xmax - xmin) * side) + m.UP * (
                    yc / (ymax - ymin) * side))
                cells.add(cell)
        self.add(cells)

        # Рамка вокруг тепловой карты
        frame = m.Rectangle(width=side, height=side, color=m.WHITE, stroke_width=1)
        frame.move_to(center)
        self.add(frame)

        # Маргинальная p(x) сверху
        marg_x_axes = m.Axes(
            x_range=[-1.8, 1.8, 1],
            y_range=[0, 1.2, 0.5],
            x_length=side, y_length=1.0,
            axis_config={"color": m.GRAY}
        )
        marg_x_axes.move_to(center + m.UP * (side / 2 + 0.6))
        marg_x = marg_x_axes.plot(
            lambda x: float(np.exp(-0.5 * ((x - mu1[0]) ** 2) / Sigma[0, 0]))
            / sqrt(2 * pi * Sigma[0, 0]),
            x_range=[-1.8, 1.8], color=m.RED
        )
        self.add(marg_x_axes, marg_x)
        self.add(m.Text("p(x)", color=m.RED, font_size=18).next_to(
            marg_x_axes, m.LEFT))

        # Маргинальная p(y) справа
        marg_y_axes = m.Axes(
            x_range=[0, 1.2, 0.5],
            y_range=[-1.8, 1.8, 1],
            x_length=1.0, y_length=side,
            axis_config={"color": m.GRAY}
        )
        marg_y_axes.move_to(center + m.RIGHT * (side / 2 + 0.6))
        marg_y = marg_y_axes.plot(
            lambda y: float(np.exp(-0.5 * ((y - mu1[1]) ** 2) / Sigma[1, 1]))
            / sqrt(2 * pi * Sigma[1, 1]),
            x_range=[-1.8, 1.8], color=m.GREEN
        )
        self.add(marg_y_axes, marg_y)
        self.add(m.Text("p(y)", color=m.GREEN, font_size=18).next_to(
            marg_y_axes, m.UP))

        # Вертикальная полоса x = x0
        x0 = 0.6
        vline = m.Line(
            center + m.RIGHT * (x0 / (xmax - xmin) * side) + m.DOWN * (side / 2),
            center + m.RIGHT * (x0 / (xmax - xmin) * side) + m.UP * (side / 2),
            color=m.WHITE, stroke_width=2
        )
        self.add(vline)
        self.add(m.MathTex("x_0", color=m.WHITE, font_size=22).next_to(
            vline, m.UP, buff=0.1))

        # Условное p(y|x0) - отдельный график
        cond_axes = m.Axes(
            x_range=[0, 1.2, 0.5],
            y_range=[-1.8, 1.8, 1],
            x_length=1.2, y_length=2.0,
            axis_config={"color": m.GRAY}
        )
        cond_axes.to_corner(m.DR).shift(m.LEFT * 0.4 + m.UP * 1.2)

        mu_cond = mu1[1] + Sigma[1, 0] / Sigma[0, 0] * (x0 - mu1[0])
        var_cond = Sigma[1, 1] - Sigma[1, 0] ** 2 / Sigma[0, 0]
        cond_curve = cond_axes.plot(
            lambda y: float(_norm_pdf(y, mu_cond, sqrt(var_cond))),
            x_range=[-1.8, 1.8], color=m.ORANGE
        )
        self.add(cond_axes, cond_curve)
        self.add(m.MathTex(r"p(y\mid x_0)", color=m.ORANGE, font_size=20).next_to(
            cond_axes, m.UP))

        title = m.Text("Совместное → маргинальные и условное",
                       font_size=22, color=m.WHITE).to_edge(m.UP)
        self.add(title)
        self.wait(2)


# ============================================================
# 4. Преобразование с якобианом: (x1, x2) → (x1, x2²)
# ============================================================
class JacobianTransform(Scene_):
    """
    Равномерное облако точек в [0,1]² сжимается вниз преобразованием y2=x2².
    """
    def construct(self):
        np.random.seed(3)
        n_pts = 400
        xs = np.random.uniform(0, 1, n_pts)
        ys = np.random.uniform(0, 1, n_pts)

        left_plane = m.NumberPlane(
            x_range=[0, 1, 0.25], y_range=[0, 1, 0.25],
            x_length=3.2, y_length=3.2,
            background_line_style={"stroke_opacity": 0.3, "stroke_color": m.GRAY}
        ).shift(m.LEFT * 3.2)

        right_plane = left_plane.copy().shift(m.RIGHT * 6.4)
        self.add(left_plane, right_plane)

        left_dots = m.VGroup()
        for i in range(n_pts):
            left_dots.add(m.Dot(
                left_plane.coords_to_point(xs[i], ys[i]),
                color=m.BLUE, radius=0.02, fill_opacity=0.7
            ))
        self.add(left_dots)

        # Преобразованные точки
        right_dots = m.VGroup()
        for i in range(n_pts):
            right_dots.add(m.Dot(
                right_plane.coords_to_point(xs[i], ys[i] ** 2),
                color=m.RED, radius=0.02, fill_opacity=0.7
            ))

        label_left = m.Text("Равномерно в [0,1]²",
                            font_size=20).next_to(left_plane, m.DOWN)
        self.add(label_left)

        self.play(m.Transform(left_dots.copy(), right_dots), run_time=2)
        self.add(right_dots)

        label_right = m.MathTex(r"y_1=x_1,\ y_2=x_2^2",
                                font_size=22, color=m.RED).next_to(
            right_plane, m.DOWN)
        self.add(label_right)

        jac = m.MathTex(
            r"|J| = \left|\frac{\partial y}{\partial x}\right| = 2x_2",
            font_size=30, color=m.YELLOW
        ).to_edge(m.UP)
        self.add(jac)

        caption = m.Text("Плотность растёт там, где |J| мало",
                         font_size=22, color=m.YELLOW).to_edge(m.DOWN)
        self.add(caption)
        self.wait(2)


# ============================================================
# 5. Разная дисперсия при одинаковом среднем
# ============================================================
class VarianceComparison(Scene_):
    """
    Два нормальных распределения с одинаковым μ и разными σ.
    """
    def construct(self):
        axes = m.Axes(
            x_range=[-5, 5, 1],
            y_range=[0, 1.0, 0.2],
            x_length=9, y_length=5,
            axis_config={"color": m.GRAY}
        )
        axes.add_coordinates()
        self.add(axes)

        curve_narrow = axes.plot(
            lambda x: _norm_pdf(x, 0, 0.7),
            x_range=[-5, 5], color=m.BLUE
        )
        curve_wide = axes.plot(
            lambda x: _norm_pdf(x, 0, 1.8),
            x_range=[-5, 5], color=m.RED
        )
        self.add(curve_narrow, curve_wide)

        label1 = m.MathTex(r"\sigma = 0.7", color=m.BLUE,
                           font_size=26).to_corner(m.UL)
        label2 = m.MathTex(r"\sigma = 1.8", color=m.RED,
                           font_size=26).next_to(label1, m.DOWN, aligned_edge=m.LEFT)

        mu_line = m.DashedLine(
            axes.coords_to_point(0, 0),
            axes.coords_to_point(0, 1.0),
            color=m.YELLOW
        )
        mu_label = m.MathTex(r"\mu = 0", color=m.YELLOW,
                             font_size=22).next_to(mu_line, m.UR)

        self.add(label1, label2, mu_line, mu_label)

        caption = m.Text("Одинаковое среднее — разная дисперсия",
                         font_size=22, color=m.WHITE).to_edge(m.DOWN)
        self.add(caption)
        self.wait(2)


# ============================================================
# 6. Шесть облаков точек с разной корреляцией
# ============================================================
class CorrelationScatter(Scene_):
    """
    Шесть scatter-plot'ов: ρ = −1, −0.7, 0, 0.3, 0.7, 1.
    """
    def construct(self):
        np.random.seed(7)
        rhos = [-1.0, -0.7, 0.0, 0.3, 0.7, 1.0]
        n_pts = 120

        positions = [
            m.UP * 1.7 + m.LEFT * 4.2,
            m.UP * 1.7 + m.LEFT * 0.1,
            m.UP * 1.7 + m.RIGHT * 4.0,
            m.DOWN * 1.7 + m.LEFT * 4.2,
            m.DOWN * 1.7 + m.LEFT * 0.1,
            m.DOWN * 1.7 + m.RIGHT * 4.0,
        ]

        size = 2.6

        for rho, pos in zip(rhos, positions):
            # Оси
            plane = m.NumberPlane(
                x_range=[-3, 3, 1], y_range=[-3, 3, 1],
                x_length=size, y_length=size,
                background_line_style={"stroke_opacity": 0.15},
                axis_config={"stroke_width": 1}
            ).move_to(pos)

            # Генерация коррелированных точек
            cov = np.array([[1, rho], [rho, 1]])
            try:
                L = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                L = np.array([[1, 0], [rho, sqrt(max(0, 1 - rho**2))]])
            pts = np.random.randn(n_pts, 2) @ L.T

            dots = m.VGroup(*[
                m.Dot(plane.coords_to_point(p[0], p[1]),
                      color=m.BLUE, radius=0.025, fill_opacity=0.7)
                for p in pts
            ])

            label = m.MathTex(rf"\rho = {rho}", font_size=22,
                              color=m.YELLOW).next_to(plane, m.UP, buff=0.15)

            self.add(plane, dots, label)

        caption = m.Text("С ростом |ρ| облако сжимается в линию",
                         font_size=22, color=m.WHITE).to_edge(m.DOWN)
        self.add(caption)
        self.wait(2)


# ============================================================
# 7. Три бета-плотности
# ============================================================
class BetaDistribution(Scene_):
    """
    Бета-плотности с разными (a, b).
    """
    def construct(self):
        axes = m.Axes(
            x_range=[0, 1, 0.2],
            y_range=[0, 4, 1],
            x_length=9, y_length=5,
            axis_config={"color": m.GRAY}
        )
        axes.add_coordinates()
        self.add(axes)

        curve1 = axes.plot(lambda x: float(_beta_pdf(x, 2, 2)),
                           x_range=[0.001, 0.999], color=m.BLUE)
        curve2 = axes.plot(lambda x: float(_beta_pdf(x, 5, 1)),
                           x_range=[0.001, 0.999], color=m.GREEN)
        curve3 = axes.plot(lambda x: float(_beta_pdf(x, 0.5, 0.5)),
                           x_range=[0.001, 0.999], color=m.RED)
        self.add(curve1, curve2, curve3)

        label1 = m.MathTex(r"(a=2,\ b=2)", color=m.BLUE,
                           font_size=24).to_corner(m.UL)
        label2 = m.MathTex(r"(a=5,\ b=1)", color=m.GREEN,
                           font_size=24).next_to(label1, m.DOWN, aligned_edge=m.LEFT)
        label3 = m.MathTex(r"(a=0.5,\ b=0.5)", color=m.RED,
                           font_size=24).next_to(label2, m.DOWN, aligned_edge=m.LEFT)
        self.add(label1, label2, label3)

        caption = m.Text("Форма беты управляется числом наблюдений",
                         font_size=22, color=m.WHITE).to_edge(m.DOWN)
        self.add(caption)
        self.wait(2)


# ============================================================
# 8. Тяжёлые хвосты: нормальное, t(3), Парето
# ============================================================
class HeavyTails(Scene_):
    """
    Хвосты на логарифмической оси Y.
    """
    def construct(self):
        axes = m.Axes(
            x_range=[0, 6, 1],
            y_range=[-8, 0.5, 1],  # log10
            x_length=9, y_length=5.5,
            axis_config={"color": m.GRAY}
        )
        axes.add_coordinates()
        self.add(axes)

        def log_norm(x):
            return np.log10(np.maximum(_norm_pdf(x, 0, 1), 1e-10))

        def log_t(x):
            return np.log10(np.maximum(_t_pdf(x, 3), 1e-10))

        def log_pareto(x):
            return np.log10(np.maximum(_pareto_pdf(x, 1.0, 1.5), 1e-10))

        # Сдвигаем нормальное и t, чтобы хвосты были видны для x>0
        # Обычное N(0,1) быстро уходит вниз, t медленнее, Парето пологая.
        g_norm = axes.plot(log_norm, x_range=[0.1, 6], color=m.BLUE)
        g_t = axes.plot(log_t, x_range=[0.1, 6], color=m.GREEN)
        g_pareto = axes.plot(log_pareto, x_range=[1.01, 6], color=m.RED)

        self.add(g_norm, g_t, g_pareto)

        label_norm = m.Text("Нормальное",
                            color=m.BLUE, font_size=22).to_corner(m.UL)
        label_t = m.Text("t (ν=3)",
                         color=m.GREEN, font_size=22).next_to(label_norm, m.DOWN, aligned_edge=m.LEFT)
        label_p = m.Text("Парето",
                         color=m.RED, font_size=22).next_to(label_t, m.DOWN, aligned_edge=m.LEFT)
        self.add(label_norm, label_t, label_p)

        caption = m.Text("Логарифмическая ось Y: тяжесть хвоста",
                         font_size=22, color=m.WHITE).to_edge(m.DOWN)
        self.add(caption)
        self.wait(2)


# ============================================================
# 9. ЦПТ: гистограммы выборочного среднего
# ============================================================
class CentralLimitTheorem(Scene_):
    """
    Три гистограммы среднего: n=1, 5, 30. Наложена кривая нормальной плотности.
    """
    def construct(self):
        np.random.seed(11)
        n_samples = 3000

        def make_hist_axes(pos, x_len=2.8, y_len=2.0):
            ax = m.Axes(
                x_range=[0, 1, 0.5],
                y_range=[0, 1, 0.5],
                x_length=x_len, y_length=y_len,
                axis_config={"color": m.GRAY, "stroke_width": 1}
            ).move_to(pos)
            return ax

        positions = [m.LEFT * 4.2, m.ORIGIN, m.RIGHT * 4.2]

        for n, pos in zip([1, 5, 30], positions):
            means = np.mean(np.random.uniform(0, 1, size=(n_samples, n)), axis=1)
            # Гистограмма
            bins = np.linspace(0, 1, 21)
            counts, _ = np.histogram(means, bins=bins, density=True)
            max_c = max(counts.max(), 1e-6)

            ax = make_hist_axes(pos)
            # Нормируем высоту так, чтобы макс ~ 1
            for i in range(len(counts)):
                h = counts[i] / max_c
                if h < 1e-4:
                    continue
                rect = m.Rectangle(
                    width=2.8 / 20 * 0.95,
                    height=2.0 * h,
                    fill_opacity=0.7,
                    stroke_width=0
                )
                rect.set_fill(m.BLUE)
                x_center = (bins[i] + bins[i + 1]) / 2
                rect.move_to(ax.coords_to_point(x_center, h / 2))
                self.add(rect)

            # Кривая нормальной плотности
            mu = 0.5
            sigma = sqrt((1 / 12) / n)
            norm_curve = ax.plot(
                lambda x: float(_norm_pdf(x, mu, sigma)) / max_c,
                x_range=[max(0.001, mu - 4 * sigma), min(0.999, mu + 4 * sigma)],
                color=m.YELLOW
            )
            self.add(ax, norm_curve)

            label = m.MathTex(rf"n = {n}", font_size=24,
                              color=m.WHITE).next_to(ax, m.UP, buff=0.2)
            self.add(label)

        caption = m.Text("ЦПТ: с ростом n среднее становится нормальным",
                         font_size=22, color=m.WHITE).to_edge(m.DOWN)
        self.add(caption)
        self.wait(2)


# ============================================================
# 10. Кросс-энтропия: истинное vs предсказанное
# ============================================================
class CrossEntropy(Scene_):
    """
    Два столбца: истинное (one-hot) и предсказанное распределение.
    """
    def construct(self):
        # Истинное распределение — класс «кот»
        true_labels = ["кот", "собака", "лиса"]
        true_probs = [1.0, 0.0, 0.0]
        pred_probs = [0.2, 0.5, 0.3]

        # Позиции столбцов
        centers = [m.LEFT * 3.5, m.LEFT * 1.2, m.RIGHT * 1.1]
        base_y = -2.0
        max_h = 3.5

        # Подписи классов
        for c, lbl in zip(centers, true_labels):
            self.add(m.Text(lbl, font_size=22).move_to(c + m.DOWN * 2.4))

        # Истинные столбцы
        for c, p in zip(centers, true_probs):
            rect = m.Rectangle(
                width=0.9, height=p * max_h,
                fill_opacity=0.8, stroke_width=1, color=m.WHITE
            )
            rect.set_fill(m.GREEN)
            rect.move_to(c + m.UP * (base_y + p * max_h / 2) + m.RIGHT * 0)
            self.add(rect)
            if p > 0:
                self.add(m.MathTex(f"{p:.1f}", font_size=20).next_to(
                    rect, m.UP, buff=0.1))

        # Предсказанные столбцы (смещены вправо на 0.6)
        for c, p in zip(centers, pred_probs):
            rect = m.Rectangle(
                width=0.9, height=p * max_h,
                fill_opacity=0.8, stroke_width=1, color=m.WHITE
            )
            rect.set_fill(m.ORANGE)
            rect.move_to(c + m.UP * (base_y + p * max_h / 2) + m.RIGHT * 1.0)
            self.add(rect)
            self.add(m.MathTex(f"{p:.1f}", font_size=20).next_to(
                rect, m.UP, buff=0.1))

        # Заголовки
        self.add(m.Text("Истинное", font_size=22, color=m.GREEN).move_to(
            m.LEFT * 2.3 + m.UP * 2.0))
        self.add(m.Text("Предсказание", font_size=22, color=m.ORANGE).move_to(
            m.RIGHT * 0.4 + m.UP * 2.0))

        # Формула
        ce_label = m.MathTex(
            r"H = -\ln(0.2) \approx 1.61",
            font_size=30, color=m.YELLOW
        ).to_corner(m.UR)
        self.add(ce_label)

        ce_label2 = m.MathTex(
            r"\text{если } p(\text{кот})=0.9:\ H \approx 0.105",
            font_size=24, color=m.GREEN
        ).next_to(ce_label, m.DOWN, aligned_edge=m.RIGHT)
        self.add(ce_label2)

        caption = m.Text("Кросс-энтропия штрафует за низкую p на истинном классе",
                         font_size=20, color=m.WHITE).to_edge(m.DOWN)
        self.add(caption)
        self.wait(2)


# ============================================================
# 11. KL-дивергенция: p и q
# ============================================================
class KLDivergence(Scene_):
    """
    Две плотности p (синяя) и q (оранжевая); заштрихована область,
    где p велико, а q мало.
    """
    def construct(self):
        axes = m.Axes(
            x_range=[-5, 5, 1],
            y_range=[0, 0.6, 0.2],
            x_length=9, y_length=5.5,
            axis_config={"color": m.GRAY}
        )
        axes.add_coordinates()
        self.add(axes)

        p = lambda x: _norm_pdf(x, -1.0, 1.0)
        q = lambda x: _norm_pdf(x, 1.2, 1.4)

        curve_p = axes.plot(p, x_range=[-5, 5], color=m.BLUE)
        curve_q = axes.plot(q, x_range=[-5, 5], color=m.ORANGE)
        self.add(curve_p, curve_q)

        # Заштрихованные области, где p > q
        xs = np.linspace(-5, 5, 300)
        # Найдём пересечения
        diff = np.array([p(x) - q(x) for x in xs])
        sign = np.sign(diff)
        # Найдём границы области p > q
        mask = diff > 0
        if mask.any():
            idx = np.where(mask)[0]
            # Возьмём непрерывный участок
            x_lo = xs[idx[0]]
            x_hi = xs[idx[-1]]
            xs_sh = np.linspace(x_lo, x_hi, 60)
            pts = [axes.coords_to_point(x, 0.0) for x in xs_sh]
            pts += [axes.coords_to_point(x, float(p(x))) for x in reversed(xs_sh)]
            shade = m.Polygon(*pts, color=m.RED, fill_opacity=0.3, stroke_width=0)
            self.add(shade)

            # Вертикальные линии
            for xb in [x_lo, x_hi]:
                self.add(m.DashedLine(
                    axes.coords_to_point(xb, 0),
                    axes.coords_to_point(xb, float(p(xb))),
                    color=m.RED, stroke_width=1
                ))

        label_p = m.MathTex("p(x)", color=m.BLUE, font_size=26).to_corner(m.UL)
        label_q = m.MathTex("q(x)", color=m.ORANGE, font_size=26).next_to(
            label_p, m.DOWN, aligned_edge=m.LEFT)
        self.add(label_p, label_q)

        kl_formula = m.MathTex(
            r"D_{KL}(p\,\|\,q) = \int p(x)\ln\frac{p(x)}{q(x)}\,dx",
            font_size=28, color=m.YELLOW
        ).to_corner(m.UR)
        self.add(kl_formula)

        caption = m.Text(
            'KL штрафует за то, что q "забыл" о массе p',
            font_size=22, color=m.WHITE
        ).to_edge(m.DOWN)
        self.add(caption)
        self.wait(2)


# ============================================================
# 1. Нормальное распределение: изменение дисперсии
# ============================================================
class NormalVariance(Scene_):
    """
    Колокол нормального распределения. Дисперсия меняется: при увеличении — расползается,
    при уменьшении — сужается. Исходный колокол остаётся бледной тенью.
    """
    def construct(self):
        axes = m.Axes(
            x_range=[-6, 6, 1], y_range=[0, 0.9, 0.2],
            x_length=10, y_length=4.5, axis_config={"color": m.GRAY}
        )
        axes.add_coordinates()
        self.add(axes)

        mu = 0.0
        sigma_tracker = m.ValueTracker(1.0)

        # Бледная тень исходного колокола (σ = 1)
        ghost = axes.plot(lambda x: np.exp(-x**2/2)/np.sqrt(2*np.pi),
                          x_range=[-6, 6], color=m.GRAY, stroke_opacity=0.4)
        self.add(ghost, m.MathTex(r"\sigma^2=1", color=m.GRAY,
                                  font_size=18).next_to(ghost.get_end(), m.UR))

        # Живая кривая
        def pdf(x):
            s2 = sigma_tracker.get_value()
            return np.exp(-x**2/(2*s2))/np.sqrt(2*np.pi*s2)

        curve = m.always_redraw(
            lambda: axes.plot(pdf, x_range=[-6, 6], color=m.BLUE, stroke_width=3)
        )
        self.add(curve)

        # Среднее
        mean_line = axes.get_vertical_line(axes.c2p(0, 0), color=m.RED, stroke_width=2)
        mean_label = m.MathTex(r"\mu = 0", color=m.RED, font_size=24).next_to(
            axes.c2p(0, 0.75), m.UP)
        self.add(mean_line, mean_label)

        # Показ дисперсии
        var_value = m.DecimalNumber(1.0, num_decimal_places=2,
                                    color=m.YELLOW, font_size=34)
        var_value.add_updater(lambda d: d.set_value(sigma_tracker.get_value()))
        var_label = m.MathTex(r"\sigma^2 = ", color=m.WHITE, font_size=30)
        var_group = m.VGroup(var_label, var_value).arrange(m.RIGHT, buff=0.15)
        var_group.to_corner(m.UP + m.RIGHT, buff=0.5)
        self.add(var_group)

        title = m.Text("Нормальное распределение: изменение дисперсии",
                       font_size=24, color=m.WHITE).to_edge(m.UP, buff=0.3)
        self.add(title)

        self.wait(0.5)
        self.play(sigma_tracker.animate.set_value(4.0), run_time=2)   # расширение
        self.wait(0.5)
        self.play(sigma_tracker.animate.set_value(0.3), run_time=2)   # сжатие
        self.wait(0.5)
        self.play(sigma_tracker.animate.set_value(2.0), run_time=2)   # среднее значение
        self.wait(2)


# ============================================================
# 2. Scatter + регрессия + коэффициент корреляции
# ============================================================
class ScatterCorrelation(Scene_):
    """
    Облако точек вокруг прямой. Коэффициент корреляции r меняется:
    при уменьшении облако расширяется, при росте — точки жмутся к прямой.
    """
    def construct(self):
        np.random.seed(42)
        n = 200
        x = np.random.uniform(-3, 3, n)

        r_tracker = m.ValueTracker(0.9)

        axes = m.Axes(
            x_range=[-4, 4, 1], y_range=[-4, 4, 1],
            x_length=7, y_length=7, axis_config={"color": m.GRAY}
        )
        axes.add_coordinates()
        axes.shift(m.LEFT * 1.2)
        self.add(axes)

        # Кривая регрессии: y = x
        reg_line = axes.plot(lambda xv: xv, x_range=[-4, 4], color=m.RED, stroke_width=3)
        self.add(reg_line)

        # Scatter, зависящий от r
        def make_dots():
            r = r_tracker.get_value()
            noise_std = np.sqrt(np.clip((1 - r**2) / r**2, 0.001, 10)) * 1.5
            y = x + np.random.normal(0, noise_std, n)
            return m.VGroup(*[
                m.Dot(axes.c2p(x[i], y[i]), radius=0.045,
                      color=m.BLUE, fill_opacity=0.6)
                for i in range(n)
            ])

        dots = m.always_redraw(make_dots)
        self.add(dots)

        # Показ r
        r_value = m.DecimalNumber(0.9, num_decimal_places=2,
                                  color=m.YELLOW, font_size=36)
        r_value.add_updater(lambda d: d.set_value(r_tracker.get_value()))
        r_label = m.MathTex(r"r = ", color=m.WHITE, font_size=32)
        r_group = m.VGroup(r_label, r_value).arrange(m.RIGHT, buff=0.15)
        r_group.to_corner(m.UP + m.RIGHT, buff=0.6)
        self.add(r_group)

        title = m.Text("Корреляция и разброс вокруг регрессии",
                       font_size=24, color=m.WHITE).to_edge(m.UP, buff=0.3)
        self.add(title)

        self.wait(0.5)
        self.play(r_tracker.animate.set_value(0.2), run_time=2)    # расширение
        self.wait(0.5)
        self.play(r_tracker.animate.set_value(0.98), run_time=2)   # сжатие
        self.wait(0.5)
        self.play(r_tracker.animate.set_value(0.7), run_time=2)
        self.wait(2)


# ============================================================
# 3. Бета-распределение: изменение параметров
# ============================================================
class BetaDistribution(Scene_):
    """
    Плотность Beta(a, b). Меняем b → 1, затем возвращаем, меняем a → 0.5,
    затем оба → 0.5. В конце — несколько кривых с подписями.
    """
    def construct(self):
        axes = m.Axes(
            x_range=[0, 1, 0.2], y_range=[0, 5, 1],
            x_length=9, y_length=5, axis_config={"color": m.GRAY}
        )
        axes.add_coordinates()
        self.add(axes)

        a_tracker = m.ValueTracker(2.0)
        b_tracker = m.ValueTracker(2.0)

        def beta_pdf(x):
            a, b = a_tracker.get_value(), b_tracker.get_value()
            x = np.clip(x, 1e-3, 1 - 1e-3)
            # log-версия для устойчивости
            from scipy.special import betaln
            logp = (a - 1) * np.log(x) + (b - 1) * np.log(1 - x) - betaln(a, b)
            return np.exp(logp)

        curve = m.always_redraw(
            lambda: axes.plot(beta_pdf, x_range=[0.005, 0.995],
                              color=m.BLUE, stroke_width=3)
        )
        self.add(curve)

        # Параметры
        a_val = m.DecimalNumber(2.0, num_decimal_places=1, color=m.GREEN, font_size=30)
        b_val = m.DecimalNumber(2.0, num_decimal_places=1, color=m.RED, font_size=30)
        a_val.add_updater(lambda d: d.set_value(a_tracker.get_value()))
        b_val.add_updater(lambda d: d.set_value(b_tracker.get_value()))
        a_label = m.MathTex(r"a = ", color=m.GREEN, font_size=28)
        b_label = m.MathTex(r"b = ", color=m.RED, font_size=28)
        a_grp = m.VGroup(a_label, a_val).arrange(m.RIGHT, buff=0.1)
        b_grp = m.VGroup(b_label, b_val).arrange(m.RIGHT, buff=0.1)
        params = m.VGroup(a_grp, b_grp).arrange(m.DOWN, buff=0.2)
        params.to_corner(m.UP + m.RIGHT, buff=0.5)
        self.add(params)

        title = m.Text("Бета-распределение Beta(a, b)",
                       font_size=24, color=m.WHITE).to_edge(m.UP, buff=0.3)
        self.add(title)

        # Сохраняем промежуточные кривые для финала
        saved_curves = []

        # Шаг 1: b → 1
        self.wait(0.3)
        self.play(b_tracker.animate.set_value(1.0), run_time=2)
        self.wait(0.3)
        saved_curves.append(axes.plot(beta_pdf, x_range=[0.005, 0.995],
                                       color=m.RED, stroke_width=2, stroke_opacity=0.5))

        # Шаг 2: b → 2, a → 0.5
        self.play(b_tracker.animate.set_value(2.0),
                  a_tracker.animate.set_value(0.5), run_time=2)
        self.wait(0.3)
        saved_curves.append(axes.plot(beta_pdf, x_range=[0.005, 0.995],
                                       color=m.GREEN, stroke_width=2, stroke_opacity=0.5))

        # Шаг 3: a = b = 0.5
        self.play(b_tracker.animate.set_value(0.5), run_time=2)
        self.wait(0.3)
        saved_curves.append(axes.plot(beta_pdf, x_range=[0.005, 0.995],
                                       color=m.PURPLE, stroke_width=2, stroke_opacity=0.5))

        # Показываем все сохранённые кривые на финальном кадре
        for c in saved_curves:
            self.add(c)

        # Легенда
        legend = m.VGroup(
            m.Text("b=1", color=m.RED, font_size=18),
            m.Text("a=0.5", color=m.GREEN, font_size=18),
            m.Text("a=b=0.5", color=m.PURPLE, font_size=18)
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.15)
        legend.to_corner(m.DOWN + m.LEFT, buff=0.5)
        self.add(legend)
        self.wait(2)


# ============================================================
# 4. Хвосты распределений: нормальное, t(3), Парето
# ============================================================
class HeavyTails(Scene_):
    """
    Плотности: N(0,1), t с ν=3, Парето. Видны разные скорости спада хвостов.
    """
    def construct(self):
        axes = m.Axes(
            x_range=[0, 6, 1], y_range=[0, 1.2, 0.2],
            x_length=10, y_length=5, axis_config={"color": m.GRAY}
        )
        axes.add_coordinates()
        self.add(axes)

        normal_curve = axes.plot(
            lambda x: 2 * stats.norm.pdf(x),  # умножаем на 2, чтобы лучше видеть
            x_range=[0.001, 6], color=m.BLUE, stroke_width=3
        )
        t_curve = axes.plot(
            lambda x: 2 * stats.t.pdf(x, df=3),
            x_range=[0.001, 6], color=m.ORANGE, stroke_width=3
        )
        pareto_curve = axes.plot(
            lambda x: 2 * stats.pareto.pdf(x, b=0.3, scale=0.5),
            x_range=[0.51, 6], color=m.RED, stroke_width=3
        )

        n_lbl = m.MathTex(r"\mathcal{N}(0,1)", color=m.BLUE, font_size=24).next_to(normal_curve.get_start(), m.UR)
        t_lbl = m.MathTex(r"t_{\nu=3}", color=m.ORANGE, font_size=28).next_to(t_curve.get_start(), m.DR)
        p_lbl = m.Text("Парето", color=m.RED, font_size=24).next_to(pareto_curve.get_end(), m.UR)

        self.play(
            m.Create(normal_curve), m.Write(n_lbl),
            m.Create(t_curve), m.Write(t_lbl),
            m.Create(pareto_curve), m.Write(p_lbl),
            run_time=2
        )

        title = m.Text("Хвосты распределений (правая полуось)",
                       font_size=24, color=m.WHITE).to_edge(m.UP, buff=0.3)
        self.add(title)
        note = m.Text("Нормальное спадает быстрее всех, Парето — медленнее всех",
                      font_size=20, color=m.YELLOW).to_edge(m.DOWN, buff=0.4)
        self.add(note)
        self.wait(3)


# ============================================================
# 5. ЦПТ: гистограммы выборочного среднего для n=1,5,30
# ============================================================
class CLTHistograms(Scene_):
    """
    Последовательность гистограмм среднего для n=1, 5, 30.
    Исходное — равномерное U(0,1). Наложена кривая нормальной плотности.
    """
    def construct(self):
        np.random.seed(0)

        def make_axes(shift):
            ax = m.Axes(
                x_range=[0, 1, 0.2], y_range=[0, 4, 1],
                x_length=4, y_length=3, axis_config={"color": m.GRAY}
            )
            ax.shift(shift)
            return ax

        positions = [m.LEFT * 4.2, m.ORIGIN, m.RIGHT * 4.2]
        titles = ["n = 1", "n = 5", "n = 30"]

        axes_list = [make_axes(p) for p in positions]
        for ax in axes_list:
            self.add(ax)

        title_text = m.Text("Центральная предельная теорема: средние выборок из U(0,1)",
                            font_size=22, color=m.WHITE).to_edge(m.UP, buff=0.3)
        self.add(title_text)

        for i, (n, ax, pos) in enumerate(zip([1, 5, 30], axes_list, positions)):
            samples = np.random.uniform(0, 1, (2000, n)).mean(axis=1)
            counts, bins = np.histogram(samples, bins=25, range=(0, 1))
            hist = ax.get_vertical_bar_chart(values=counts,
                bar_names=[f"{bins[i]:.1f}" for i in range(len(bins)-1)],
                color=[m.BLUE, m.GREEN, m.RED][i], fill_opacity=0.6
            )
            self.play(m.Create(hist), run_time=1.5)

            # Наложение нормальной плотности
            mu, sig = 0.5, np.sqrt(1/12/n)
            x_vals = np.linspace(0, 1, 100)
            y_vals = norm_dist.pdf(x_vals, mu, sig)
            # нормируем под плотность гистограммы (по оси x шкала [0,1], 25 бинов)
            # plot_histogram в manim нормирует по умолчанию нет — просто покажем общий контур
            curve = ax.plot(lambda x: norm_dist.pdf(x, mu, sig) * (2000 * 1/25),
                            x_range=[max(0.01, mu - 4*sig), min(0.99, mu + 4*sig)],
                            color=m.YELLOW, stroke_width=3)
            self.add(curve)

            lbl = m.Text(titles[i], font_size=24, color=m.YELLOW).next_to(ax, m.UP, buff=0.2)
            self.add(lbl)
            self.wait(0.5)
            self.remove(hist, lbl, curve)

        note = m.Text("С ростом n распределение среднего стремится к нормальному",
                      font_size=18, color=m.YELLOW).to_edge(m.DOWN, buff=0.4)
        self.add(note)
        self.wait(3)


# ============================================================
# 6. KL-дивергенция
# ============================================================
class KLDivergence(Scene_):
    """
    Две плотности p (синяя) и q (оранжевая). Заштрихован вклад в KL:
    где p велико, а q мало. Кривые двигаются, KL пересчитывается.
    """
    def construct(self):
        axes = m.Axes(
            x_range=[-6, 6, 1], y_range=[0, 0.6, 0.1],
            x_length=10, y_length=4.5, axis_config={"color": m.GRAY}
        )
        axes.add_coordinates()
        self.add(axes)

        mu_p = -0.5
        sigma_p = 1.0

        mu_q_tracker = m.ValueTracker(1.0)
        sigma_q_tracker = m.ValueTracker(1.0)

        def p_pdf(x):
            return stats.norm.pdf(x, mu_p, sigma_p)

        def q_pdf(x):
            return stats.norm.pdf(x, mu_q_tracker.get_value(),
                                  sigma_q_tracker.get_value())

        p_curve = axes.plot(p_pdf, x_range=[-6, 6], color=m.BLUE, stroke_width=3)
        q_curve = m.always_redraw(
            lambda: axes.plot(q_pdf, x_range=[-6, 6], color=m.ORANGE, stroke_width=3)
        )
        self.add(p_curve, q_curve)

        # Штриховка разности
        def make_shade():
            # Заштриховываем область где p>q (вклад p log(p/q))
            x_vals = np.linspace(-6, 6, 300)
            pv = p_pdf(x_vals)
            qv = q_pdf(x_vals)
            diff = np.clip(pv - qv, 0, None)
            # создаём многоугольники для каждого бина
            polys = m.VGroup()
            for i in range(len(x_vals) - 1):
                if diff[i] > 1e-4 and diff[i+1] > 1e-4:
                    x0, x1 = x_vals[i], x_vals[i+1]
                    y0, y1 = pv[i], pv[i+1]
                    # вертикальная полоска между кривыми
                    poly = m.Polygon(
                        axes.c2p(x0, qv[i]),
                        axes.c2p(x1, qv[i+1]),
                        axes.c2p(x1, pv[i+1]),
                        axes.c2p(x0, pv[i]),
                        fill_color=m.YELLOW, fill_opacity=0.35,
                        stroke_width=0
                    )
                    polys.add(poly)
            return polys

        shade = m.always_redraw(make_shade)
        self.add(shade)

        # KL значение
        def compute_kl():
            x_vals = np.linspace(-8, 8, 500)
            pv = p_pdf(x_vals)
            qv = np.clip(q_pdf(x_vals), 1e-12, None)
            kl = np.trapezoid(pv * np.log(pv / qv), x_vals)
            return kl

        kl_val = m.DecimalNumber(compute_kl(), num_decimal_places=3,
                                 color=m.YELLOW, font_size=34)
        kl_val.add_updater(lambda d: d.set_value(compute_kl()))
        kl_lbl = m.MathTex(r"D_{KL}(p \| q) \approx ", color=m.WHITE, font_size=28)
        kl_grp = m.VGroup(kl_lbl, kl_val).arrange(m.RIGHT, buff=0.15)
        kl_grp.to_corner(m.UP + m.RIGHT, buff=0.5).shift(m.DOWN)
        self.add(kl_grp)

        title = m.Text("KL-дивергенция: штраф за «забытую» массу p",
                       font_size=24, color=m.WHITE).to_edge(m.UP, buff=0.3)
        self.add(title)

        p_lbl = m.MathTex(r"p(x)", color=m.BLUE, font_size=24).next_to(p_curve.get_start(), m.UR)
        q_lbl = m.MathTex(r"q(x)", color=m.ORANGE, font_size=24).next_to(q_curve.get_end(), m.UR)
        self.add(p_lbl, q_lbl)

        note = m.Text("KL штрафует за то, что q «забыл» о массе p",
                      font_size=18, color=m.YELLOW).to_edge(m.DOWN, buff=0.4)
        self.add(note)

        # Анимации
        self.wait(0.5)
        self.play(mu_q_tracker.animate.set_value(2.5), run_time=2)
        self.wait(0.3)
        self.play(mu_q_tracker.animate.set_value(-0.5),
                  sigma_q_tracker.animate.set_value(2.5), run_time=2)
        self.wait(0.3)
        self.play(mu_q_tracker.animate.set_value(0.3),
                  sigma_q_tracker.animate.set_value(0.7), run_time=2)
        self.wait(2)


# ============================================================
# 7. Смещённая / несмещённая / эффективная оценки
# ============================================================
class EstimatorComparison(Scene_):
    """
    Три плотности распределения оценок:
    1) смещённая (центр не в 0),
    2) несмещённая, но с большой дисперсией,
    3) несмещённая и эффективная (малая дисперсия).
    """
    def construct(self):
        axes = m.Axes(
            x_range=[-6, 6, 1], y_range=[0, 0.6, 0.1],
            x_length=10, y_length=4.5, axis_config={"color": m.GRAY}
        )
        axes.add_coordinates()
        self.add(axes)

        # Истинное значение
        true_line = axes.get_vertical_line(axes.c2p(0, 0), color=m.RED, stroke_width=3)
        true_lbl = m.MathTex(r"\theta", color=m.RED, font_size=28).next_to(
            axes.c2p(0, 0.55), m.UP)
        self.add(true_line, true_lbl)

        # Смещённая
        biased = axes.plot(lambda x: stats.norm.pdf(x, 2.5, 0.8),
                           x_range=[-6, 6], color=m.ORANGE, stroke_width=3)
        # Несмещённая с большой дисперсией
        unbiased_high = axes.plot(lambda x: stats.norm.pdf(x, 0, 2.0),
                                  x_range=[-6, 6], color=m.BLUE, stroke_width=3)
        # Эффективная
        efficient = axes.plot(lambda x: stats.norm.pdf(x, 0, 0.6),
                              x_range=[-6, 6], color=m.GREEN, stroke_width=3)

        lbl1 = m.Text("Смещённая", color=m.ORANGE, font_size=20).shift(m.RIGHT * 2)
        lbl2 = m.Text("Несмещённая,\nбольшая дисперсия", color=m.BLUE, font_size=18).shift(m.LEFT * 2 + m.DOWN)
        lbl3 = m.Text("Несмещённая\nи эффективная", color=m.GREEN, font_size=18).shift(m.LEFT * 2 + m.UP)

        self.play(
            m.Create(biased), m.Write(lbl1),
            m.Create(unbiased_high), m.Write(lbl2),
            m.Create(efficient), m.Write(lbl3),
            run_time=2
        )

        title = m.Text("Сравнение оценок: смещение и эффективность",
                       font_size=24, color=m.WHITE).to_edge(m.UP, buff=0.3)
        self.add(title)
        note = m.Text("Хорошая оценка: несмещённая (центр в θ) и с малой дисперсией",
                      font_size=18, color=m.YELLOW).to_edge(m.DOWN, buff=0.4)
        self.add(note)
        self.wait(3)


# ============================================================
# 8. Эмпирическая CDF vs истинная
# ============================================================
class EmpiricalCDF(Scene_):
    """
    Истинная CDF F(t) (синяя гладкая) и эмпирическая F̂ₙ(t) (оранжевая ступенчатая).
    С ростом n ступеньки становятся мельче, кривые сливаются.
    """
    def construct(self):
        np.random.seed(42)
        axes = m.Axes(
            x_range=[-4, 4, 1], y_range=[0, 1, 0.2],
            x_length=10, y_length=5, axis_config={"color": m.GRAY}
        )
        axes.add_coordinates()
        self.add(axes)

        # Истинная CDF N(0,1)
        true_cdf = axes.plot(lambda x: stats.norm.cdf(x),
                             x_range=[-4, 4], color=m.BLUE, stroke_width=3)
        self.add(true_cdf)
        self.add(m.MathTex(r"F(t)", color=m.BLUE, font_size=24).next_to(true_cdf.get_end(), m.UR))

        n_tracker = m.ValueTracker(20)

        def make_ecdf():
            n = int(n_tracker.get_value())
            samples = np.random.normal(0, 1, n)
            samples_sorted = np.sort(samples)
            # Рисуем ступенчатую функцию
            points = [axes.c2p(-4, 0)]
            for i, s in enumerate(samples_sorted):
                points.append(axes.c2p(s, i/n))
                points.append(axes.c2p(s, (i+1)/n))
            points.append(axes.c2p(4, 1))
            return m.VMobject().set_points_as_corners(
                [p for p in points]
            ).set_stroke(color=m.ORANGE, width=3)

        ecdf = m.always_redraw(make_ecdf)
        self.add(ecdf)
        ecdf_lbl = m.MathTex(r"\hat{F}_n(t)", color=m.ORANGE, font_size=24).next_to(
            axes.c2p(3, 0.3), m.UR)
        self.add(ecdf_lbl)

        # Значение n
        n_val = m.Integer(n_tracker.get_value(), color=m.YELLOW, font_size=32)
        n_val.add_updater(lambda d: d.set_value(int(n_tracker.get_value())))
        n_lbl = m.MathTex(r"n = ", color=m.WHITE, font_size=28)
        n_grp = m.VGroup(n_lbl, n_val).arrange(m.RIGHT, buff=0.15)
        n_grp.to_corner(m.UP + m.RIGHT, buff=0.5)
        self.add(n_grp)

        title = m.Text("Эмпирическая CDF сходится к истинной",
                       font_size=24, color=m.WHITE).to_edge(m.UP, buff=0.3)
        self.add(title)
        note = m.Text("С ростом n ступеньки мельчают и кривые сливаются",
                      font_size=18, color=m.YELLOW).to_edge(m.DOWN, buff=0.4)
        self.add(note)

        self.wait(0.5)
        self.play(n_tracker.animate.set_value(5), run_time=1.5)
        self.wait(0.5)
        self.play(n_tracker.animate.set_value(100), run_time=2)
        self.wait(0.5)
        self.play(n_tracker.animate.set_value(500), run_time=2)
        self.wait(2)


# ============================================================
# 9. Геометрия L1 vs L2 регуляризации
# ============================================================
class L1vsL2(Scene_):
    """
    Две панели: эллипсы MSE и уровни штрафа (круг для L2, ромб для L1).
    Точка касания: для L1 решение на оси (вес = 0), для L2 — нет.
    """
    def construct(self):
        # ---------- Левая панель: L2 ----------
        ax_l2 = m.Axes(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1],
            x_length=4.5, y_length=4.5, axis_config={"color": m.GRAY}
        ).shift(m.LEFT * 3.5)
        self.add(ax_l2)

        # Эллипсы MSE (центр смещён)
        for r in [0.5, 1.0, 1.5]:
            ellipse = m.Ellipse(width=2*r*0.7, height=2*r*1.1,
                                color=m.BLUE, stroke_width=2)
            ellipse.move_to(ax_l2.c2p(0.8, 0.8))
            self.add(ellipse)

        # L2-шар (окружность)
        l2_circle = m.Circle(radius=1.0, color=m.GREEN, stroke_width=3)
        l2_circle.move_to(ax_l2.c2p(0, 0))
        self.add(l2_circle)

        # Точка касания (примерно)
        touch_l2 = m.Dot(ax_l2.c2p(0.7, 0.7), color=m.RED, radius=0.1)
        self.add(touch_l2)
        self.add(m.Text("L2: точка касания\nне на оси",
                        color=m.WHITE, font_size=18).next_to(ax_l2, m.DOWN, buff=0.3))
        self.add(m.Text("L2 (Ridge)", color=m.GREEN, font_size=22).next_to(ax_l2, m.UP, buff=0.3))

        # ---------- Правая панель: L1 ----------
        ax_l1 = m.Axes(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1],
            x_length=4.5, y_length=4.5, axis_config={"color": m.GRAY}
        ).shift(m.RIGHT * 3.5)
        self.add(ax_l1)

        for r in [0.5, 1.0, 1.5]:
            ellipse = m.Ellipse(width=2*r*0.7, height=2*r*1.1,
                                color=m.BLUE, stroke_width=2)
            ellipse.move_to(ax_l1.c2p(0.8, 0.8))
            self.add(ellipse)

        # L1-шар (ромб)
        l1_diamond = m.Polygon(
            ax_l1.c2p(1, 0), ax_l1.c2p(0, 1),
            ax_l1.c2p(-1, 0), ax_l1.c2p(0, -1),
            color=m.GREEN, stroke_width=3, fill_opacity=0.1
        )
        self.add(l1_diamond)

        # Точка касания — на оси
        touch_l1 = m.Dot(ax_l1.c2p(1, 0), color=m.RED, radius=0.1)
        self.add(touch_l1)
        self.add(m.Text("L1: точка касания\nна оси (вес = 0)",
                        color=m.WHITE, font_size=18).next_to(ax_l1, m.DOWN, buff=0.3))
        self.add(m.Text("L1 (Lasso)", color=m.GREEN, font_size=22).next_to(ax_l1, m.UP, buff=0.3))

        title = m.Text("Геометрия регуляризации", font_size=24, color=m.WHITE).to_edge(m.UP, buff=0.2)
        self.add(title)
        self.wait(3)


# ============================================================
# 10. Ошибки I и II рода
# ============================================================
class TypeIandII(Scene_):
    """
    Два пересекающихся колокола: H0 (синий) и H1 (оранжевый).
    Правый хвост H0 — α (ложная тревога).
    Левый хвост H1 — β (пропуск эффекта).
    """
    def construct(self):
        axes = m.Axes(
            x_range=[-5, 8, 1], y_range=[0, 0.5, 0.1],
            x_length=10, y_length=4.5, axis_config={"color": m.GRAY}
        )
        axes.add_coordinates()
        self.add(axes)

        mu0_tracker = m.ValueTracker(0.0)
        mu1_tracker = m.ValueTracker(3.0)

        def h0_pdf(x):
            return stats.norm.pdf(x, mu0_tracker.get_value(), 1.0)

        def h1_pdf(x):
            return stats.norm.pdf(x, mu1_tracker.get_value(), 1.0)

        h0 = m.always_redraw(lambda: axes.plot(h0_pdf, x_range=[-5, 8],
                                                color=m.BLUE, stroke_width=3))
        h1 = m.always_redraw(lambda: axes.plot(h1_pdf, x_range=[-5, 8],
                                                color=m.ORANGE, stroke_width=3))
        self.add(h0, h1)

        # Порог — на середине между средними
        crit_tracker = m.ValueTracker(1.5)

        def make_alpha():
            c = crit_tracker.get_value()
            x_range = np.linspace(c, 8, 100)
            return axes.get_area(h0, x_range=[c, 8], color=m.BLUE, opacity=0.5)

        def make_beta():
            c = crit_tracker.get_value()
            return axes.get_area(h1, x_range=[-5, c], color=m.ORANGE, opacity=0.5)

        alpha_area = m.always_redraw(make_alpha)
        beta_area = m.always_redraw(make_beta)
        self.add(alpha_area, beta_area)

        crit_line = m.always_redraw(
            lambda: axes.get_vertical_line(axes.c2p(crit_tracker.get_value(), 0.4),
                                            color=m.RED, stroke_width=2)
        )
        self.add(crit_line)

        # Подписи
        lbl_h0 = m.MathTex(r"H_0", color=m.BLUE, font_size=28).next_to(
            axes.c2p(-1, 0.42), m.UP)
        lbl_h1 = m.MathTex(r"H_1", color=m.ORANGE, font_size=28).next_to(
            axes.c2p(4, 0.42), m.UP)
        self.add(lbl_h0, lbl_h1)

        # α и β как значения
        alpha_val = m.DecimalNumber(stats.norm.sf(crit_tracker.get_value()), num_decimal_places=3,
                                     color=m.BLUE, font_size=26)
        alpha_val.add_updater(lambda d: d.set_value(stats.norm.sf(crit_tracker.get_value())))
        beta_val = m.DecimalNumber(stats.norm.cdf(crit_tracker.get_value() - mu1_tracker.get_value()),
                                    num_decimal_places=3, color=m.ORANGE, font_size=26)
        beta_val.add_updater(lambda d: d.set_value(
            stats.norm.cdf(crit_tracker.get_value() - mu1_tracker.get_value())))

        a_lbl = m.MathTex(r"\alpha = ", color=m.BLUE, font_size=22)
        b_lbl = m.MathTex(r"\beta = ", color=m.ORANGE, font_size=22)
        alpha_grp = m.VGroup(a_lbl, alpha_val).arrange(m.RIGHT, buff=0.1)
        beta_grp = m.VGroup(b_lbl, beta_val).arrange(m.RIGHT, buff=0.1)
        stats_grp = m.VGroup(alpha_grp, beta_grp).arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT)
        stats_grp.to_corner(m.UP + m.RIGHT, buff=0.5)
        self.add(stats_grp)

        title = m.Text("Ошибки I и II рода", font_size=24, color=m.WHITE).to_edge(m.UP, buff=0.3)
        self.add(title)

        note = m.Text("Синий: «ложная тревога» (α)   •   Оранжевый: «пропуск эффекта» (β)",
                      font_size=18, color=m.YELLOW).to_edge(m.DOWN, buff=0.4)
        self.add(note)

        self.wait(0.5)
        self.play(mu1_tracker.animate.set_value(1.5), run_time=2)
        self.wait(0.5)
        self.play(mu1_tracker.animate.set_value(5.0), run_time=2)
        self.wait(0.5)
        self.play(crit_tracker.animate.set_value(2.0), run_time=1.5)
        self.wait(2)


# ============================================================
# 11. Три DAG: цепочка, развилка, слияние
# ============================================================
class ThreeDAGs(Scene_):
    """
    Три маленьких DAG бок о бок:
    1) Цепочка X → Z → Y — «медиатор»
    2) Развилка X ← Z → Y — «конфаундер»
    3) Слияние X → Z ← Y — «коллайдер»
    """
    def construct(self):
        def make_node(label, pos, color=m.BLUE):
            circle = m.Circle(radius=0.4, color=color, fill_opacity=0.2).move_to(pos)
            txt = m.MathTex(label, font_size=32).move_to(pos)
            return m.VGroup(circle, txt)

        def make_arrow(p1, p2):
            return m.Arrow(p1, p2, buff=0.45, color=m.GRAY, stroke_width=3)

        # --- 1. Цепочка ---
        pos_X = np.array([-5.5, -1.2, 0])
        pos_Z = np.array([-5.5, 0.0, 0])
        pos_Y = np.array([-5.5, 1.2, 0])
        X1 = make_node("X", pos_X)
        Z1 = make_node("Z", pos_Z, color=m.GREEN)
        Y1 = make_node("Y", pos_Y)
        chain = m.VGroup(
            X1, Z1, Y1,
            make_arrow(pos_X, pos_Z),
            make_arrow(pos_Z, pos_Y),
            m.Text("Медиатор", font_size=20, color=m.YELLOW).next_to(Z1, m.RIGHT, buff=0.5)
        )

        # --- 2. Развилка ---
        pos_Z2 = np.array([0, 0, 0])
        pos_X2 = np.array([-1.2, -1.5, 0])
        pos_Y2 = np.array([1.2, -1.5, 0])
        X2 = make_node("X", pos_X2)
        Z2 = make_node("Z", pos_Z2, color=m.GREEN)
        Y2 = make_node("Y", pos_Y2)
        fork = m.VGroup(
            X2, Z2, Y2,
            make_arrow(pos_Z2, pos_X2),
            make_arrow(pos_Z2, pos_Y2),
            m.Text("Конфаундер", font_size=20, color=m.YELLOW).next_to(Z2, m.UP, buff=0.3)
        )

        # --- 3. Слияние ---
        pos_X3 = np.array([4.0, -1.5, 0])
        pos_Y3 = np.array([6.4, -1.5, 0])
        pos_Z3 = np.array([5.2, 0.5, 0])
        X3 = make_node("X", pos_X3)
        Z3 = make_node("Z", pos_Z3, color=m.GREEN)
        Y3 = make_node("Y", pos_Y3)
        collider = m.VGroup(
            X3, Z3, Y3,
            make_arrow(pos_X3, pos_Z3),
            make_arrow(pos_Y3, pos_Z3),
            m.Text("Коллайдер", font_size=20, color=m.YELLOW).next_to(Z3, m.UP, buff=0.3)
        )

        self.play(m.Create(chain), m.Create(fork), m.Create(collider))

        title = m.Text("Три базовых DAG: медиатор, конфаундер, коллайдер",
                       font_size=24, color=m.WHITE).to_edge(m.UP, buff=0.4)
        self.add(title)
        self.wait(3)


# ============================================================
# 12. Дискриминативный vs генеративный подходы
# ============================================================
class DiscriminativeVsGenerative(Scene_):
    """
    Слева: облака точек двух классов + разделяющая граница (p(y|x)).
    Справа: те же точки + контуры плотностей (p(x,y)).
    """
    def construct(self):
        np.random.seed(42)
        # Класс 0 (синий) — слева-внизу
        c0 = np.random.multivariate_normal([-1.0, -0.5], [[0.6, 0.2], [0.2, 0.6]], 80)
        # Класс 1 (красный) — справа-вверху
        c1 = np.random.multivariate_normal([1.0, 0.5], [[0.6, 0.2], [0.2, 0.6]], 80)

        # Общие оси
        def make_axes(shift):
            ax = m.Axes(
                x_range=[-3, 3, 1], y_range=[-3, 3, 1],
                x_length=4.5, y_length=4.5, axis_config={"color": m.GRAY}
            )
            ax.shift(shift)
            return ax

        ax_L = make_axes(m.LEFT * 3.5)
        ax_R = make_axes(m.RIGHT * 3.5)
        self.add(ax_L, ax_R)

        # --- Левая панель: облака + граница ---
        dots_L = m.VGroup()
        for p in c0:
            dots_L.add(m.Dot(ax_L.c2p(p[0], p[1]), radius=0.05, color=m.BLUE, fill_opacity=0.7))
        for p in c1:
            dots_L.add(m.Dot(ax_L.c2p(p[0], p[1]), radius=0.05, color=m.RED, fill_opacity=0.7))
        self.add(dots_L)

        # Разделяющая граница (перпендикуляр к разности средних)
        boundary = m.Line(
            ax_L.c2p(-2.5, 1.0), ax_L.c2p(2.5, -1.0),
            color=m.YELLOW, stroke_width=3
        )
        self.add(boundary)
        lbl_L = m.Text("Дискриминативный:\nучим p(y|x) — где граница?",
                       font_size=18, color=m.WHITE).next_to(ax_L, m.DOWN, buff=0.3)
        self.add(lbl_L)

        # --- Правая панель: облака + эллипсы плотности ---
        dots_R = m.VGroup()
        for p in c0:
            dots_R.add(m.Dot(ax_R.c2p(p[0], p[1]), radius=0.05, color=m.BLUE, fill_opacity=0.7))
        for p in c1:
            dots_R.add(m.Dot(ax_R.c2p(p[0], p[1]), radius=0.05, color=m.RED, fill_opacity=0.7))
        self.add(dots_R)

        # Контуры (несколько уровней для каждой гауссианы)
        for mean, color in [([-1, -0.5], m.BLUE), ([1, 0.5], m.RED)]:
            for s in [1.0, 1.8, 2.6]:
                ell = m.Ellipse(width=s*0.9, height=s*0.9, color=color, stroke_width=2)
                ell.rotate(np.deg2rad(30))
                ell.move_to(ax_R.c2p(mean[0], mean[1]))
                self.add(ell)

        lbl_R = m.Text("Генеративный:\nучим p(x,y) — как устроены классы?",
                       font_size=18, color=m.WHITE).next_to(ax_R, m.DOWN, buff=0.3)
        self.add(lbl_R)

        title = m.Text("Дискриминативный vs генеративный подход",
                       font_size=24, color=m.WHITE).to_edge(m.UP, buff=0.3)
        self.add(title)
        self.wait(3)


# ============================================================
# 13. Латентная переменная: кластеры и z
# ============================================================
class LatentVariable(Scene_):
    """
    Облако точек из двух гауссиан, пунктирные стрелки к «латентному» z.
    """
    def construct(self):
        np.random.seed(1)
        # Три кластера
        means = [(-2, 1), (0, -1.5), (2, 0.5)]
        colors = [m.BLUE, m.GREEN, m.RED]
        groups = []
        for mean, color in zip(means, colors):
            pts = np.random.multivariate_normal(mean, [[0.3, 0], [0, 0.3]], 30)
            groups.append((pts, color))

        axes = m.Axes(
            x_range=[-4, 4, 1], y_range=[-3, 3, 1],
            x_length=8, y_length=6, axis_config={"color": m.GRAY}
        ).shift(m.LEFT * 1.5)
        self.add(axes)

        all_dots = m.VGroup()
        for pts, color in groups:
            for p in pts:
                all_dots.add(m.Dot(axes.c2p(p[0], p[1]), radius=0.06,
                                   color=color, fill_opacity=0.7))
        self.add(all_dots)
        self.add(m.Text("Наблюдаемые x", font_size=20, color=m.WHITE)
                 .next_to(axes, m.DOWN, buff=0.3))

        # Правая сторона: латентные точки z
        z_axes = m.Axes(
            x_range=[-1.5, 1.5, 1], y_range=[-1.5, 1.5, 1],
            x_length=3, y_length=3, axis_config={"color": m.GRAY}
        ).shift(m.RIGHT * 3.5)
        self.add(z_axes)

        z_positions = [(-1, 0.8), (0, -0.7), (1, 0.4)]
        z_dots = m.VGroup()
        for (zp, (pts, color)) in zip(z_positions, groups):
            z_dots.add(m.Dot(z_axes.c2p(*zp), radius=0.15, color=color))
        self.add(z_dots)
        self.add(m.Text("Латентный z (невидим)", font_size=20, color=m.YELLOW)
                 .next_to(z_axes, m.DOWN, buff=0.3))

        # Пунктирные стрелки от облаков к z
        for (pts, color), zp in zip(groups, z_positions):
            center = pts.mean(axis=0)
            start = axes.c2p(*center)
            end = z_axes.c2p(*zp)
            arrow = m.DashedLine(start, end, color=color, stroke_width=2, dash_length=0.15)
            self.add(arrow)

        title = m.Text("Латентная переменная z порождает наблюдаемые x",
                       font_size=24, color=m.WHITE).to_edge(m.UP, buff=0.3)
        self.add(title)
        self.wait(3)


# ============================================================
# 14. MCMC на «банане»
# ============================================================
class MCMCbanana(Scene_):
    """
    Двумерное «банановое» распределение. По нему ползёт марковская цепь:
    зелёные стрелки — принятые шаги, красные — отвергнутые.
    """
    def construct(self):
        np.random.seed(7)

        title = m.Text("MCMC на произвольном распределении",
                       font_size=24, color=m.WHITE).to_edge(m.UP, buff=0.3)
        self.add(title)
        note = m.Text("Зелёные — принятые шаги, серые — отвергнутые",
                      font_size=18, color=m.YELLOW).to_edge(m.DOWN, buff=0.4)
        self.add(note)

        axes = m.Axes(
            x_range=[-4, 4, 1], y_range=[-4, 4, 1],
            x_length=8, y_length=8, axis_config={"color": m.GRAY}
        )
        self.add(axes)

        # Банановая плотность
        def banana_logp(x, y):
            # нелинейное преобразование
            return -0.5 * (x**2 / 2 + (y - 0.3*x**2 + 1)**2 / 0.5)

        # Показываем контуры плотности
        for lvl in [0.3, 0.6, 1.0, 1.5, 2.0]:
            # грубо рисуем контуры как эллипсы, изогнутые по y ≈ 0.3 x^2 - 1
            pts = []
            xs = np.linspace(-3.5, 3.5, 100)
            for x in xs:
                pts.append(axes.c2p(x, 0.3*x**2 - 1 + lvl))
            # используем точки как контур
            curve = m.VMobject().set_points_as_corners(pts).set_stroke(m.BLUE, width=2, opacity=0.4)
            self.add(curve)
            pts2 = []
            for x in xs:
                pts2.append(axes.c2p(x, 0.3*x**2 - 1 - lvl))
            curve2 = m.VMobject().set_points_as_corners(pts2).set_stroke(m.BLUE, width=2, opacity=0.4)
            self.add(curve2)

        # MCMC — Metropolis-Hastings (без анимации каждого шага, чтобы не затягивать)
        # Готовим последовательность, потом проигрываем быстро
        n_steps = 100
        x_cur = np.array([0.0, 0.0])
        chain = [x_cur.copy()]
        accepted = []
        for _ in range(n_steps):
            x_prop = x_cur + np.random.normal(0, 0.5, 2)
            lp_cur = banana_logp(*x_cur)
            lp_prop = banana_logp(*x_prop)
            if np.log(np.random.uniform()) < lp_prop - lp_cur:
                accepted.append(True)
                x_cur = x_prop
            else:
                accepted.append(False)
            chain.append(x_cur.copy())

        # Рисуем последовательно с анимацией
        prev_point = None
        trajectory_dots = m.VGroup()
        for i, (pt, acc) in enumerate(zip(chain, accepted + [True])):
            dot = m.Dot(axes.c2p(*pt), radius=0.04,
                        color=m.GREEN if acc else m.RED, fill_opacity=0.7)
            trajectory_dots.add(dot)
            self.play(m.Create(dot), run_time=0.1)
            if prev_point is not None and acc:
                arrow = m.Line(
                    axes.c2p(*prev_point), axes.c2p(*pt),
                    color=m.GREEN, stroke_width=2, stroke_opacity=0.6
                )
                trajectory_dots.add(arrow)
                self.play(m.Create(arrow), run_time=0.1)
            prev_point = pt
        self.add(trajectory_dots)
        self.wait(3)


# ============================================================
# 15. Схема VAE
# ============================================================
class VAEDiagram(Scene_):
    """
    Схема VAE: x → encoder → (μ, σ) → μ + σ·ε → z → decoder → x̂.
    Отдельная стрелка от шума ε ∼ N(0, I).
    """
    def construct(self):
        def make_box(label, pos, width=1.5, height=0.8, color=m.BLUE):
            box = m.RoundedRectangle(width=width, height=height,
                                     color=color, stroke_width=2,
                                     fill_color=color, fill_opacity=0.15).move_to(pos)
            txt = m.Text(label, font_size=20, color=m.WHITE).move_to(pos)
            return m.VGroup(box, txt)

        y = 0.5
        # Основная цепочка
        x_box = make_box("x", np.array([-6, y, 0]), color=m.GRAY)
        enc = make_box("Encoder", np.array([-4, y, 0]), width=1.8, color=m.BLUE)
        mu_box = make_box("μ, σ", np.array([-2, y, 0]), color=m.GREEN)
        rep = make_box("μ + σ·ε", np.array([0, y, 0]), width=1.6, color=m.PURPLE)
        z_box = make_box("z", np.array([2, y, 0]), color=m.GREEN)
        dec = make_box("Decoder", np.array([4, y, 0]), width=1.8, color=m.BLUE)
        xhat = make_box("x̂", np.array([6.0, y, 0]), color=m.GRAY)

        boxes = m.VGroup(x_box, enc, mu_box, rep, z_box, dec, xhat)
        self.add(boxes)

        # Стрелки
        arrows_positions = [
            (np.array([-4.75, y, 0]), np.array([-4.4, y, 0])),
            (np.array([-2.6, y, 0]), np.array([-2.1, y, 0])),
            (np.array([-0.3, y, 0]), np.array([0.4, y, 0])),
            (np.array([2.0, y, 0]), np.array([2.4, y, 0])),
            (np.array([3.95, y, 0]), np.array([4.1, y, 0])),
            (np.array([5.9, y, 0]), np.array([6.05, y, 0])),
        ]
        # for p1, p2 in arrows_positions:
        #     self.add(m.Arrow(p1, p2, buff=0.05, color=m.GRAY, stroke_width=3))

        # Стрелка от шума ε
        eps_box = make_box("ε ~ N(0, I)", np.array([0, y - 1.7, 0]),
                           width=2.0, color=m.YELLOW)
        self.add(eps_box)
        self.add(m.Arrow(eps_box.get_top(), rep.get_bottom(),
                         buff=0.1, color=m.YELLOW, stroke_width=3))

        # Подпись про репараметризацию
        note = m.Text("Градиент идёт сквозь z благодаря репараметризации",
                      font_size=20, color=m.YELLOW).to_edge(m.DOWN, buff=0.5)
        self.add(note)

        title = m.Text("Схема VAE (Variational Autoencoder)",
                       font_size=24, color=m.WHITE).to_edge(m.UP, buff=0.3)
        self.add(title)
        self.wait(3)


# ============================================================
# 16. Связь концепций: данные → модель → вывод → предсказания → калибровка
# ============================================================
class ConceptMap(Scene_):
    """
    Схема связи концепций: данные → вероятностная модель → апостериорный вывод
    → предсказания с неопределённостью → калибровка/конформные интервалы.
    """
    def construct(self):
        def make_box(text, pos, color=m.BLUE, width=4.0):
            box = m.RoundedRectangle(width=width, height=0.9,
                                     color=color, stroke_width=2,
                                     fill_color=color, fill_opacity=0.15).move_to(pos)
            txt = m.Text(text, font_size=20, color=m.WHITE).move_to(pos)
            return m.VGroup(box, txt)

        y = 0.0
        dx = 1.33
        positions = [
            np.array([y, 2*dx, 0]),
            np.array([y, dx, 0]),
            np.array([y, 0, 0]),
            np.array([y, -dx, 0]),
            np.array([y, -2*dx, 0]),
        ]
        labels = [
            "Данные",
            "Вероятностная\nмодель",
            "Апостериорный вывод\n(точный/прибл.)",
            "Предсказания с\nнеопределённостью",
            "Калибровка,\nконформные интервалы",
        ]
        colors = [m.GRAY, m.BLUE, m.GREEN, m.PURPLE, m.YELLOW]

        boxes = m.VGroup(*[make_box(l, p, c) for l, p, c in zip(labels, positions, colors)])
        self.add(boxes)

        # Стрелки
        for i in range(len(boxes) - 1):
            p1 = boxes[i].get_bottom()
            p2 = boxes[i+1].get_top()
            self.add(m.Arrow(p1, p2, buff=0.1, color=m.GRAY, stroke_width=3))

        title = m.Text("Связь концепций: от данных к надёжным предсказаниям",
                       font_size=24, color=m.WHITE).to_edge(m.UP, buff=0.5)
        self.add(title)

        note = m.Text("Единая цепочка байесовского моделирования",
                      font_size=20, color=m.YELLOW).to_edge(m.DOWN, buff=0.5)
        self.add(note)
        self.wait(3)


class CrossEntropyIntuition(Scene_):
    """
    Кросс-энтропия H(p, q) = -Σ p(x) log q(x).
    Слева — истинное распределение p (бины-столбики).
    Посередине — предсказание модели q.
    Справа — вклад в кросс-энтропию: -p(x) log q(x).
    При «плохом» q (высокая уверенность не в том классе) штраф резко растёт.
    Показаны значения H(p,q) и H(p) (энтропия).
    """
    def construct(self):
        # -------- Истинное распределение p (4 класса) --------
        p = np.array([0.1, 0.7, 0.1, 0.1])   # истинный класс — 2-й (индекс 1)
        labels = ["A", "B", "C", "D"]

        # Трекер для q — «уверенность» модели
        q_b_tracker = m.ValueTracker(0.7)  # значение для правильного класса B

        def make_q():
            # Остальные три класса делят оставшуюся массу поровну
            b = np.clip(q_b_tracker.get_value(), 1e-3, 1 - 1e-3)
            rest = (1 - b) / 3
            return np.array([rest, b, rest, rest])

        # -------- Оси и бары --------
        def make_axes(shift):
            ax = m.Axes(
                x_range=[0, 4, 1], y_range=[0, 1, 0.25],
                x_length=3.5, y_length=2.8,
                axis_config={"color": m.GRAY}
            ).shift(shift)
            return ax

        ax_p = make_axes(m.LEFT * 4.5 + m.UP * 1.2)
        ax_q = make_axes(m.LEFT * 0.3 + m.UP * 1.2)
        ax_ce = make_axes(m.RIGHT * 4.0 + m.UP * 1.2)
        for ax in [ax_p, ax_q, ax_ce]:
            self.add(ax)

        # Подписи осей
        self.add(m.MathTex(r"p(x)", color=m.BLUE, font_size=28)
                 .next_to(ax_p, m.UP, buff=0.15))
        self.add(m.MathTex(r"q(x)", color=m.ORANGE, font_size=28)
                 .next_to(ax_q, m.UP, buff=0.15))
        self.add(m.MathTex(r"-p(x)\log q(x)", color=m.RED, font_size=22)
                 .next_to(ax_ce, m.UP, buff=0.15))

        # Метки классов
        for ax in [ax_p, ax_q, ax_ce]:
            for i, lbl in enumerate(labels):
                self.add(m.Text(lbl, font_size=16, color=m.GRAY)
                         .next_to(ax.c2p(i + 0.5, 0), m.DOWN, buff=0.1))

        # -------- Бары p (статические) --------
        for i, v in enumerate(p):
            bar = m.Rectangle(
                width=0.7, height=v * 2.8,
                fill_color=m.BLUE, fill_opacity=0.7, stroke_width=0
            )
            bar.move_to(ax_p.c2p(i + 0.5, 0), aligned_edge=m.DOWN)
            self.add(bar)

        # -------- Бары q (динамические) --------
        def make_q_bars():
            q = make_q()
            bars = m.VGroup()
            for i, v in enumerate(q):
                bar = m.Rectangle(
                    width=0.7, height=max(v * 2.8, 0.001),
                    fill_color=m.ORANGE, fill_opacity=0.7, stroke_width=0
                )
                bar.move_to(ax_q.c2p(i + 0.5, 0), aligned_edge=m.DOWN)
                bars.add(bar)
            return bars

        q_bars = m.always_redraw(make_q_bars)
        self.add(q_bars)

        # -------- Вклад в кросс-энтропию (динамический) --------
        def make_ce_bars():
            q = make_q()
            bars = m.VGroup()
            for i in range(4):
                contrib = -p[i] * np.log(max(q[i], 1e-6))
                bar = m.Rectangle(
                    width=0.7, height=max(contrib * 2.8, 0.001),
                    fill_color=m.RED, fill_opacity=0.6, stroke_width=0
                )
                bar.move_to(ax_ce.c2p(i + 0.5, 0), aligned_edge=m.DOWN)
                bars.add(bar)
            return bars

        ce_bars = m.always_redraw(make_ce_bars)
        self.add(ce_bars)

        # -------- Численные значения H(p), H(p,q) --------
        def compute_Hp():
            return -np.sum(p * np.log(np.clip(p, 1e-12, None)))

        def compute_Hpq():
            q = make_q()
            return -np.sum(p * np.log(np.clip(q, 1e-12, None)))

        Hp_val = m.DecimalNumber(compute_Hp(), num_decimal_places=3,
                                 color=m.BLUE, font_size=26)
        Hpq_val = m.DecimalNumber(compute_Hpq(), num_decimal_places=3,
                                  color=m.RED, font_size=26)
        Hpq_val.add_updater(lambda d: d.set_value(compute_Hpq()))

        Hp_lbl = m.MathTex(r"H(p) = ", color=m.BLUE, font_size=22)
        Hpq_lbl = m.MathTex(r"H(p, q) = ", color=m.RED, font_size=22)
        Hp_grp = m.VGroup(Hp_lbl, Hp_val).arrange(m.RIGHT, buff=0.1)
        Hpq_grp = m.VGroup(Hpq_lbl, Hpq_val).arrange(m.RIGHT, buff=0.1)
        stats = m.VGroup(Hp_grp, Hpq_grp).arrange(m.DOWN, buff=0.25, aligned_edge=m.LEFT)
        stats.to_edge(m.DOWN, buff=1.0)
        self.add(stats)

        # -------- Заголовок и пояснение --------
        title = m.Text("Кросс-энтропия: штраф за уверенную ошибку",
                       font_size=24, color=m.WHITE).to_edge(m.UP, buff=0.3)
        self.add(title)
        note = m.Text(
            "Если модель уверена не в том классе — -log q(x) резко растёт",
            font_size=18, color=m.YELLOW
        ).to_edge(m.DOWN, buff=0.3)
        self.add(note)

        self.wait(0.5)

        # Сценарий 1: q уверена в правильном классе (низкий штраф)
        self.play(q_b_tracker.animate.set_value(0.95), run_time=2)
        self.wait(0.5)

        # Сценарий 2: q равномерна (штраф = log 4 ≈ 1.386)
        self.play(q_b_tracker.animate.set_value(0.25), run_time=2)
        self.wait(0.5)

        # Сценарий 3: q уверена в НЕПРАВИЛЬНОМ классе (большой штраф)
        # эмулируем — уменьшаем q_b и «уводим» массу в первый класс
        # (в этой упрощённой модели мы просто делаем q_b маленьким,
        #  что даст рост штрафа по правильному классу)
        self.play(q_b_tracker.animate.set_value(0.02), run_time=2)
        self.wait(2)


# ============================================================
# 2. Совместное, маргинальное и условное распределения
# ============================================================
class MarginalConditional(Scene_):
    """
    Тепловая карта p(x,y). Сверху — маргинальное p(x) (сумма по столбцам).
    Справа — маргинальное p(y) (сумма по строкам).
    Вертикальная полоса при x = x₀ выделяет условное p(y | x₀),
    которое строится отдельным графиком и меняется при сдвиге полосы.
    """
    def construct(self):
        np.random.seed(3)

        # -------- Сетка для совместной плотности (2D гауссиана с корреляцией) --------
        nx, ny = 40, 30
        x_vals = np.linspace(-3, 3, nx)
        y_vals = np.linspace(-3, 3, ny)
        X, Y = np.meshgrid(x_vals, y_vals)
        mu_x, mu_y = 0.3, -0.2
        sig_x, sig_y = 1.1, 0.9
        rho = 0.6
        Z = np.exp(
            -1 / (2 * (1 - rho ** 2)) * (
                ((X - mu_x) / sig_x) ** 2
                + ((Y - mu_y) / sig_y) ** 2
                - 2 * rho * (X - mu_x) * (Y - mu_y) / (sig_x * sig_y)
            )
        )
        Z /= Z.sum()

        # -------- Позиционирование --------
        # Тепловая карта (главная панель)
        cell_w = 0.14
        cell_h = 0.14
        heatmap_center = m.LEFT * 1.5 + m.DOWN * 0.3

        # Маргинал p(x) — сверху
        marg_x_ax = m.Axes(
            x_range=[-3, 3, 1], y_range=[0, 0.5, 0.25],
            x_length=nx * cell_w, y_length=1.5,
            axis_config={"color": m.GRAY}
        ).move_to(heatmap_center + m.UP * 3.4)
        # Маргинал p(y) — справа
        marg_y_ax = m.Axes(
            x_range=[0, 0.5, 0.25], y_range=[-3, 3, 1],
            x_length=1.5, y_length=ny * cell_h,
            axis_config={"color": m.GRAY}
        ).move_to(heatmap_center + m.RIGHT * 4.2)

        # -------- Рисуем тепловую карту --------
        # cmap = m.color_gradient([m.BLUE_E, m.BLUE, m.TEAL, m.YELLOW, m.RED])
        max_z = Z.max()

        heatmap_cells = m.VGroup()
        for j in range(ny):
            for i in range(nx):
                val = Z[j, i] / max_z
                color = m.interpolate_color(m.BLUE_E, m.RED, val)
                cell = m.Rectangle(
                    width=cell_w, height=cell_h,
                    fill_color=color, fill_opacity=0.85,
                    stroke_width=0
                )
                # j идёт снизу вверх
                cell.move_to(
                    heatmap_center + np.array([
                        (i - nx / 2 + 0.5) * cell_w,
                        (j - ny / 2 + 0.5) * cell_h, 0
                    ])
                )
                heatmap_cells.add(cell)
        self.add(heatmap_cells)

        # Оси тепловой карты
        hm_left = heatmap_center + np.array([-nx / 2 * cell_w, -ny / 2 * cell_h, 0])
        hm_right = heatmap_center + np.array([nx / 2 * cell_w, -ny / 2 * cell_h, 0])
        hm_top = heatmap_center + np.array([-nx / 2 * cell_w, ny / 2 * cell_h, 0])
        hm_bottom = heatmap_center + np.array([nx / 2 * cell_w, -ny / 2 * cell_h, 0])
        self.add(m.Line(hm_left, hm_right, color=m.WHITE, stroke_width=2))
        self.add(m.Line(hm_top, hm_bottom, color=m.WHITE, stroke_width=2))
        # Подписи осей тепловой карты
        self.add(m.MathTex("x", color=m.WHITE, font_size=22)
                 .next_to(hm_right, m.DOWN, buff=0.15))
        self.add(m.MathTex("y", color=m.WHITE, font_size=22)
                 .next_to(hm_top, m.LEFT, buff=0.15))

        # -------- Маргиналы --------
        marg_x = Z.sum(axis=0)  # по столбцам (по y)
        marg_y = Z.sum(axis=1)  # по строкам (по x)
        # нормируем для отображения
        marg_x_disp = marg_x / marg_x.max() * 0.45
        marg_y_disp = marg_y / marg_y.max() * 0.45

        # p(x) — бары, выровненные по верху тепловой карты
        top_y = heatmap_center[1] + ny / 2 * cell_h
        marg_x_bars = m.VGroup()
        for i in range(nx):
            h = marg_x_disp[i]
            bar = m.Rectangle(
                width=cell_w * 0.95, height=h,
                fill_color=m.TEAL, fill_opacity=0.9, stroke_width=0
            )
            bar.move_to(np.array([
                heatmap_center[0] + (i - nx / 2 + 0.5) * cell_w,
                top_y, 0
            ]), aligned_edge=m.DOWN)
            marg_x_bars.add(bar)
        self.add(marg_x_bars)

        # p(y) — бары, выровненные по правому краю тепловой карты
        right_x = heatmap_center[0] + nx / 2 * cell_w
        marg_y_bars = m.VGroup()
        for j in range(ny):
            h = marg_y_disp[j]
            bar = m.Rectangle(
                width=h, height=cell_h * 0.95,
                fill_color=m.TEAL, fill_opacity=0.9, stroke_width=0
            )
            bar.move_to(np.array([
                right_x,
                heatmap_center[1] + (j - ny / 2 + 0.5) * cell_h, 0
            ]), aligned_edge=m.LEFT)
            marg_y_bars.add(bar)
        self.add(marg_y_bars)

        # Подписи маргиналов
        self.add(m.MathTex(r"p(x)", color=m.TEAL, font_size=22)
                 .next_to(marg_x_bars, m.LEFT, buff=0.15))
        self.add(m.MathTex(r"p(y)", color=m.TEAL, font_size=22)
                 .next_to(marg_y_bars, m.UP, buff=0.15))

        # -------- Условное распределение p(y | x₀) --------
        # Сдвигаемая вертикальная полоса
        x0_tracker = m.ValueTracker(0.0)  # значение x₀ в диапазоне [-3, 3]

        # Полоса-подсветка на тепловой карте
        def make_strip():
            x0 = x0_tracker.get_value()
            i = int((x0 - (-3)) / 6 * nx)
            i = np.clip(i, 0, nx - 1)
            x_screen = heatmap_center[0] + (i - nx / 2 + 0.5) * cell_w
            strip = m.Rectangle(
                width=cell_w * 1.3, height=ny * cell_h,
                fill_color=m.YELLOW, fill_opacity=0.35,
                stroke_color=m.YELLOW, stroke_width=3
            ).move_to(np.array([x_screen, heatmap_center[1], 0]))
            return strip

        strip = m.always_redraw(make_strip)
        self.add(strip)

        # График условной плотности p(y | x0) — справа снизу
        cond_ax = m.Axes(
            x_range=[0, 0.6, 0.3], y_range=[-3, 3, 1],
            x_length=1.8, y_length=3.0,
            axis_config={"color": m.GRAY}
        ).shift(m.RIGHT * 4.0)
        self.add(cond_ax)

        def make_cond_curve():
            x0 = x0_tracker.get_value()
            i = int((x0 - (-3)) / 6 * nx)
            i = np.clip(i, 0, nx - 1)
            prof = Z[:, i]
            prof = prof / (prof.max() + 1e-12) * 0.55
            # строим кривую как замкнутую
            pts = [cond_ax.c2p(0, -3)]
            for j, v in enumerate(prof):
                y = y_vals[j]
                pts.append(cond_ax.c2p(v, y))
            pts.append(cond_ax.c2p(0, 3))
            return m.VMobject().set_points_as_corners(pts).set_stroke(
                color=m.YELLOW, width=3
            ).set_fill(color=m.YELLOW, opacity=0.3)

        cond_curve = m.always_redraw(make_cond_curve)
        self.add(cond_curve)

        # Подписи к условному графику
        self.add(m.MathTex(r"p(y\,|\,x_0)", color=m.YELLOW, font_size=20)
                 .next_to(cond_ax, m.UP, buff=0.15))
        x0_lbl = m.MathTex(r"x_0", color=m.YELLOW, font_size=20)
        x0_lbl.add_updater(lambda mob: mob.next_to(
            cond_ax, m.DOWN, buff=0.15
        ))
        self.add(x0_lbl)

        # -------- Заголовки и пояснения --------
        title = m.Text("Совместное, маргинальное и условное распределения",
                       font_size=24, color=m.WHITE).to_edge(m.UP, buff=0.3)
        self.add(title)

        note = m.Text(
            "p(x) — сумма по столбцам, p(y) — по строкам, p(y|x₀) — профиль полосы x₀",
            font_size=16, color=m.YELLOW
        ).to_edge(m.DOWN, buff=0.3)
        self.add(note)

        self.wait(0.5)

        # -------- Анимация: сдвигаем полосу --------
        self.play(x0_tracker.animate.set_value(1.5), run_time=2)
        self.wait(0.4)
        self.play(x0_tracker.animate.set_value(-1.2), run_time=2)
        self.wait(0.4)
        self.play(x0_tracker.animate.set_value(0.3), run_time=2)
        self.wait(2)


class MarginalConditionalV2(Scene_):
    """
    Тепловая карта p(x,y). Сверху — маргинальное p(x) (сумма по столбцам),
    справа — p(y) (сумма по строкам). Внутри — сдвигаемая полоса x = x₀,
    справа снизу — нормированный профиль p(y|x₀).
    """
    def construct(self):
        np.random.seed(3)

        # ---------- Параметры сетки ----------
        nx, ny = 36, 28
        x_vals = np.linspace(-3, 3, nx)
        y_vals = np.linspace(-3, 3, ny)
        X, Y = np.meshgrid(x_vals, y_vals)

        # Коррелированная 2D-гауссиана
        mu_x, mu_y = 0.4, -0.2
        sig_x, sig_y = 1.1, 0.9
        rho = 0.6
        Z = np.exp(
            -1 / (2 * (1 - rho**2)) * (
                ((X - mu_x) / sig_x)**2
                + ((Y - mu_y) / sig_y)**2
                - 2 * rho * (X - mu_x) * (Y - mu_y) / (sig_x * sig_y)
            )
        )
        Z /= Z.sum()
        max_z = Z.max()

        # ---------- Геометрия ----------
        cell_w, cell_h = 0.13, 0.13
        hm_center = m.LEFT * 1.5 + m.DOWN * 0.3
        hm_width = nx * cell_w
        hm_height = ny * cell_h
        hm_left_x = hm_center[0] - hm_width / 2
        hm_right_x = hm_center[0] + hm_width / 2
        hm_bottom_y = hm_center[1] - hm_height / 2
        hm_top_y = hm_center[1] + hm_height / 2

        # ---------- Тепловая карта ----------
        heatmap = m.VGroup()
        for j in range(ny):
            for i in range(nx):
                v = Z[j, i] / max_z
                color = m.interpolate_color(m.BLUE_E, m.RED, v)
                cell = m.Rectangle(
                    width=cell_w, height=cell_h,
                    fill_color=color, fill_opacity=0.9, stroke_width=0
                )
                cell.move_to(np.array([
                    hm_center[0] + (i - nx / 2 + 0.5) * cell_w,
                    hm_center[1] + (j - ny / 2 + 0.5) * cell_h,
                    0
                ]))
                heatmap.add(cell)
        self.add(heatmap)

        # Рамка
        frame = m.Rectangle(
            width=hm_width, height=hm_height,
            stroke_color=m.WHITE, stroke_width=2, fill_opacity=0
        ).move_to(hm_center)
        self.add(frame)

        # Подписи осей тепловой карты
        self.add(m.MathTex("x", color=m.WHITE, font_size=22)
                 .next_to(np.array([hm_right_x, hm_bottom_y, 0]), m.DOWN, buff=0.2))
        self.add(m.MathTex("y", color=m.WHITE, font_size=22)
                 .next_to(np.array([hm_left_x, hm_top_y, 0]), m.LEFT, buff=0.2))
        self.add(m.Text("совместное p(x, y)", font_size=16, color=m.LIGHT_GRAY)
                 .next_to(np.array([hm_center[0], hm_bottom_y, 0]), m.DOWN, buff=0.5))

        # ---------- Маргиналы ----------
        marg_x = Z.sum(axis=0)   # по столбцам → p(x)
        marg_y = Z.sum(axis=1)   # по строкам → p(y)
        marg_x_scale = 1.8 / marg_x.max()
        marg_y_scale = 1.8 / marg_y.max()

        # p(x) — бары сверху
        marg_x_bars = m.VGroup()
        marg_x_height = np.zeros(nx)
        for i in range(nx):
            h = marg_x[i] * marg_x_scale
            marg_x_height[i] = h
            bar = m.Rectangle(
                width=cell_w * 0.95, height=max(h, 0.001),
                fill_color=m.TEAL, fill_opacity=0.9,
                stroke_color=m.TEAL_D, stroke_width=0.5
            )
            bar.move_to(np.array([
                hm_center[0] + (i - nx / 2 + 0.5) * cell_w,
                hm_top_y + h / 2 + 0.05, 0
            ]), aligned_edge=m.DOWN)
            marg_x_bars.add(bar)

        # p(y) — бары справа
        marg_y_bars = m.VGroup()
        for j in range(ny):
            h = marg_y[j] * marg_y_scale
            bar = m.Rectangle(
                width=max(h, 0.001), height=cell_h * 0.95,
                fill_color=m.TEAL, fill_opacity=0.9,
                stroke_color=m.TEAL_D, stroke_width=0.5
            )
            bar.move_to(np.array([
                hm_right_x + h / 2 + 0.05,
                hm_center[1] + (j - ny / 2 + 0.5) * cell_h, 0
            ]), aligned_edge=m.LEFT)
            marg_y_bars.add(bar)

        # Подписи маргиналов
        lbl_px = m.MathTex(r"p(x)", color=m.TEAL, font_size=24)
        lbl_px.next_to(marg_x_bars, m.LEFT, buff=0.2)
        lbl_py = m.MathTex(r"p(y)", color=m.TEAL, font_size=24)
        lbl_py.next_to(marg_y_bars, m.UP, buff=0.2)

        # ---------- Сдвигаемая полоса ----------
        x0_tracker = m.ValueTracker(-1.2)

        def make_strip():
            x0 = x0_tracker.get_value()
            # индекс столбца
            i = int(round((x0 - (-3)) / 6 * (nx - 1)))
            i = int(np.clip(i, 0, nx - 1))
            x_screen = hm_center[0] + (i - nx / 2 + 0.5) * cell_w
            strip = m.Rectangle(
                width=cell_w * 1.3, height=hm_height,
                fill_color=m.YELLOW, fill_opacity=0.28,
                stroke_color=m.YELLOW, stroke_width=3
            ).move_to(np.array([x_screen, hm_center[1], 0]))
            return strip

        strip = m.always_redraw(make_strip)
        self.add(strip)

        # Индикатор x₀ на оси x
        def make_x0_line():
            x0 = x0_tracker.get_value()
            line = m.Line(
                np.array([hm_center[0] + (x0 / 6) * hm_width, hm_bottom_y - 0.15, 0]),
                np.array([hm_center[0] + (x0 / 6) * hm_width, hm_bottom_y, 0]),
                color=m.YELLOW, stroke_width=2
            )
            return line
        x0_marker = m.always_redraw(make_x0_line)
        self.add(x0_marker)

        x0_label = m.MathTex(r"x_0", color=m.YELLOW, font_size=20)
        x0_label.add_updater(lambda mob: mob.move_to(np.array([
            hm_center[0] + (x0_tracker.get_value() / 6) * hm_width,
            hm_bottom_y - 0.35, 0
        ])))
        self.add(x0_label)

        # ---------- Условный профиль p(y|x₀) ----------
        cond_ax = m.Axes(
            x_range=[0, 1.2, 0.5], y_range=[-3, 3, 1],
            x_length=2.2, y_length=3.4,
            axis_config={"color": m.GRAY}
        ).shift(m.RIGHT * 4.2 + m.DOWN * 1.8)
        self.add(cond_ax)

        # Нормированный профиль столбца
        def make_cond_curve():
            x0 = x0_tracker.get_value()
            i = int(round((x0 - (-3)) / 6 * (nx - 1)))
            i = int(np.clip(i, 0, nx - 1))
            prof = Z[:, i].copy()
            if prof.max() > 1e-12:
                prof = prof / prof.max() * 1.0
            # замкнутая кривая
            pts = [cond_ax.c2p(0, y_vals[0])]
            for j in range(ny):
                pts.append(cond_ax.c2p(prof[j], y_vals[j]))
            pts.append(cond_ax.c2p(0, y_vals[-1]))
            return m.VMobject().set_points_as_corners(pts).set_stroke(
                color=m.YELLOW, width=3
            ).set_fill(color=m.YELLOW, opacity=0.35)

        cond_curve = m.always_redraw(make_cond_curve)
        self.add(cond_curve)

        # Подписи условного графика
        cond_lbl = m.MathTex(r"p(y\,|\,x_0)", color=m.YELLOW, font_size=22)
        cond_lbl.next_to(cond_ax, m.UP, buff=0.2)
        self.add(cond_lbl)
        self.add(m.Text("нормированный\nпрофиль столбца",
                        font_size=14, color=m.LIGHT_GRAY)
                 .next_to(cond_ax, m.DOWN, buff=0.2))

        # ---------- Заголовок, пояснение ----------
        title = m.Text("Маргинальное vs условное распределение",
                       font_size=26, color=m.WHITE).to_edge(m.UP, buff=0.3)
        self.add(title)

        note = m.Text(
            "p(x) — сумма по столбцам;  p(y) — сумма по строкам;  p(y|x₀) — нормированный профиль полосы",
            font_size=15, color=m.YELLOW
        ).to_edge(m.DOWN, buff=0.3)
        self.add(note)

        # ---------- Вступительная анимация ----------
        self.play(
            m.FadeIn(marg_x_bars, shift=m.UP * 0.3),
            m.Write(lbl_px),
            run_time=1.5
        )
        self.play(
            m.FadeIn(marg_y_bars, shift=m.RIGHT * 0.3),
            m.Write(lbl_py),
            run_time=1.5
        )
        self.wait(0.5)

        # ---------- Движение полосы ----------
        self.play(x0_tracker.animate.set_value(1.5), run_time=2.5)
        self.wait(0.4)
        self.play(x0_tracker.animate.set_value(-2.0), run_time=2.5)
        self.wait(0.4)
        self.play(x0_tracker.animate.set_value(0.2), run_time=2.5)
        self.wait(0.4)
        self.play(x0_tracker.animate.set_value(1.0), run_time=2.5)
        self.wait(2)


if __name__ == '__main__':
    import os
    from pathlib import Path

    SCENES = [
        # "BayesTheorem",
        # "NormalIntegral",
        # "JointMarginalConditional",
        # "JacobianTransform",
        # "VarianceComparison",
        # "CorrelationScatter",
        # "BetaDistribution",
        # "HeavyTails",
        # "CentralLimitTheorem",
        # "CrossEntropy",
        # "KLDivergence",

        # "NormalVariance",
        # # "ScatterCorrelation",
        # # "BetaDistribution",
        # "HeavyTails",
        # "CLTHistograms",
        # "KLDivergence",
        # "EstimatorComparison",
        # "EmpiricalCDF",
        # # "L1vsL2",
        # "TypeIandII",
        # "ThreeDAGs",
        # # "DiscriminativeVsGenerative",
        # # "LatentVariable",
        # "MCMCbanana",
        # "VAEDiagram",
        # "ConceptMap",
        # "CrossEntropyIntuition",
        "MarginalConditional",
        # "MarginalConditionalV2",
    ]
    file_path = Path(__file__).resolve()

    for SCENE in SCENES:
        os.system(f"manim {file_path} {SCENE} -qh")
        os.system(f"manim {file_path} {SCENE} -sqh")