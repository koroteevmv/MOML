import manim as m
import numpy as np
from math import sin, ceil
from theming import LinearTransformationScene_, ThreeDScene_, Scene_, MovingCameraScene_


# ============================================================
# 1. Секущая → касательная для y = x^2
# ============================================================
class SecantToTangent(Scene_):
    """
    Точка A фиксирована, точка B приближается. Секущая AB стремится к касательной.
    """
    def construct(self):
        axes = m.Axes(
            x_range=[-1, 3, 1],
            y_range=[-1, 5, 1],
            x_length=8,
            y_length=6,
            axis_config={"color": m.GRAY}
        )
        axes.add_coordinates()
        self.add(axes)

        graph = axes.plot(lambda x: x**2, x_range=[-1, 2.5], color=m.BLUE)
        self.add(graph)

        xA = 0.5
        yA = xA**2
        dotA = m.Dot(axes.coords_to_point(xA, yA), color=m.RED)
        labelA = m.MathTex("A", color=m.RED).next_to(dotA, m.RIGHT)
        self.add(dotA, labelA)

        h = m.ValueTracker(1.5)
        xB = xA + h.get_value()
        yB = xB**2
        dotB = m.Dot(axes.coords_to_point(xB, yB), color=m.GREEN)
        labelB = m.MathTex("B", color=m.GREEN).next_to(dotB, m.RIGHT)
        self.add(dotB, labelB)

        secant = m.Line(
            axes.coords_to_point(xA, yA),
            axes.coords_to_point(xB, yB),
            color=m.YELLOW
        )
        self.add(secant)

        # Касательная в A: y = 2*xA*(x - xA) + yA = 2x - 1
        tangent = axes.plot(
            lambda x: 2 * xA * (x - xA) + yA,
            x_range=[-0.5, 2.5],
            color=m.ORANGE
        )
        self.add(tangent)

        def update_B(mob):
            xB_val = xA + h.get_value()
            yB_val = xB_val ** 2
            mob.move_to(axes.coords_to_point(xB_val, yB_val))

        def update_secant(mob):
            xB_val = xA + h.get_value()
            yB_val = xB_val ** 2
            mob.put_start_and_end_on(
                axes.coords_to_point(xA, yA),
                axes.coords_to_point(xB_val, yB_val)
            )

        dotB.add_updater(update_B)
        secant.add_updater(update_secant)

        self.play(
            h.animate.set_value(0.01),
            run_time=4,
            rate_func=m.rate_functions.ease_in_out_sine
        )

        caption = m.Text(
            "Чем меньше h, тем ближе секущая к касательной",
            font_size=24,
            color=m.YELLOW
        ).to_edge(m.DOWN)
        self.add(caption)
        self.wait(2)


# ============================================================
# 2. Экстремумы f(x)=x^3-3x и нули f'(x)=3x^2-3
# ============================================================
class ExtremaDerivative(Scene_):
    """
    Вертикальные пунктирные линии соединяют экстремумы f с нулями f'.
    """
    def construct(self):
        axes = m.Axes(
            x_range=[-3, 3, 1],
            y_range=[-5, 5, 2],
            x_length=8,
            y_length=6,
            axis_config={"color": m.GRAY}
        )
        axes.add_coordinates()
        self.add(axes)

        self.play(m.Create(f := axes.plot(lambda x: x**3 - 3 * x, x_range=[-2.5, 2.5], color=m.BLUE)))
        labelF = m.MathTex("f(x)", color=m.BLUE).shift(m.RIGHT * 3.2 + m.UP * 1)
        self.add(labelF)
        self.play(m.Create(df := axes.plot(lambda x: 3 * x**2 - 3, x_range=[-2.5, 2.5], color=m.RED)))
        labelDF = m.MathTex("f'(x)", color=m.RED).shift(m.RIGHT * 1 + m.UP * 1)
        self.add(labelDF)

        label_f = m.MathTex("f(x)=x^3-3x", color=m.BLUE).next_to(f.get_end(), m.UR)
        label_df = m.MathTex("f'(x)=3x^2-3", color=m.RED).next_to(df.get_end(), m.UR)
        self.add(f, df, label_f, label_df)

        for x0 in [-1, 1]:
            y_f = x0**3 - 3 * x0
            y_df = 0.0
            dot_f = m.Dot(axes.coords_to_point(x0, y_f), color=m.BLUE)
            dot_df = m.Dot(axes.coords_to_point(x0, y_df), color=m.RED)
            line = m.DashedLine(
                axes.coords_to_point(x0, y_f),
                axes.coords_to_point(x0, y_df),
                color=m.YELLOW
            )
            self.add(dot_f, dot_df, line)

        caption = m.Text(
            "В точках экстремума производная равна нулю",
            font_size=24,
            color=m.YELLOW
        ).to_edge(m.DOWN)
        self.add(caption)
        self.wait(2)


# ============================================================
# 3. Приближение к точке: кривая становится прямой
# ============================================================
class ZoomToTangent(MovingCameraScene_):
    """
    Камера приближается к точке на гладкой кривой — в пределе видна касательная.
    """
    def construct(self):
        axes = m.Axes(
            x_range=[-2, 4, 1],
            y_range=[-2, 4, 1],
            x_length=8,
            y_length=6,
            axis_config={"color": m.GRAY}
        )
        self.add(axes)

        func = lambda x: np.sin(x) + 1.5
        graph = axes.plot(func, x_range=[-2, 4], color=m.BLUE, stroke_width=1)
        self.add(graph)

        x0 = 1.0
        y0 = func(x0)
        # dot = m.Dot(axes.coords_to_point(x0, y0), color=m.RED)
        # self.add(dot)

        derivative = lambda x: np.cos(x)
        tangent = axes.plot(
            lambda x: derivative(x0) * (x - x0) + y0,
            x_range=[-2, 4],
            color=m.YELLOW,
            stroke_width=1
        )
        self.add(tangent)

        self.play(
            self.camera.frame.animate.scale(0.03).move_to(axes.coords_to_point(x0, y0)),
            run_time=4
        )

        caption = m.Text(
            "Приближаемся: кривая становится прямой",
            font_size=24,
            color=m.YELLOW
        ).to_edge(m.DOWN)
        self.add(caption)
        self.wait(2)


# ============================================================
# 4. Градиентный спуск по параболе L(w)=(w-3)^2
# ============================================================
class GradientDescentParabola(Scene_):
    """
    Шарик скатывается к минимуму, касательная показывает направление шага.
    """
    def construct(self):
        axes = m.Axes(
            x_range=[-1, 7, 1],
            y_range=[-1, 10, 2],
            x_length=8,
            y_length=6,
            axis_config={"color": m.GRAY}
        )
        axes.add_coordinates()
        self.add(axes)

        L = lambda w: (w - 3) ** 2 + 0.5
        dL = lambda w: 2 * (w - 3)

        graph = axes.plot(L, x_range=[0, 6], color=m.BLUE)
        self.add(graph)
        label = m.MathTex("L(w)=(w-3)^2+0.5", color=m.BLUE).next_to(graph.get_end(), m.UR).scale(0.7)
        self.add(label)

        w = m.ValueTracker(0.0)
        ball = m.Dot(
            axes.coords_to_point(w.get_value(), L(w.get_value())),
            color=m.RED,
            radius=0.1
        )
        self.add(ball)

        tangent = m.Line(
            axes.coords_to_point(w.get_value() - 1, L(w.get_value()) - dL(w.get_value())),
            axes.coords_to_point(w.get_value() + 1, L(w.get_value()) + dL(w.get_value())),
            color=m.YELLOW
        )
        self.add(tangent)

        ball.add_updater(
            lambda mob: mob.move_to(axes.coords_to_point(w.get_value(), L(w.get_value())))
        )
        tangent.add_updater(
            lambda mob: mob.put_start_and_end_on(
                axes.coords_to_point(w.get_value() - 1, L(w.get_value()) - dL(w.get_value())),
                axes.coords_to_point(w.get_value() + 1, L(w.get_value()) + dL(w.get_value()))
            )
        )

        path = m.VMobject()
        path.set_points_as_corners([axes.coords_to_point(w.get_value(), L(w.get_value()))])
        self.add(path)

        lr = 0.2
        for _ in range(10):
            w_val = w.get_value()
            grad = dL(w_val)
            w_new = w_val - lr * grad
            self.play(w.animate.set_value(w_new), run_time=0.5)
            # print((w_new, L(w_new)))
            # print(np.ndarray([axes.coords_to_point(w_new, L(w_new))]))
            # path.add_points_as_corners(axes.coords_to_point(w_new, L(w_new)))

        caption = m.Text(
            "Градиентный спуск сходится к минимуму",
            font_size=24,
            color=m.YELLOW
        ).to_edge(m.DOWN)
        self.add(caption)
        self.wait(2)


# ============================================================
# 5. ReLU, Softplus и асимптота y=x
# ============================================================
class ReLU_Softplus(Scene_):
    """
    Сравнение ReLU и её гладкой аппроксимации Softplus.
    """
    def construct(self):
        axes = m.Axes(
            x_range=[-4, 4, 1],
            y_range=[-1, 4, 1],
            x_length=8,
            y_length=6,
            axis_config={"color": m.GRAY}
        )
        axes.add_coordinates()
        self.add(axes)

        relu = axes.plot(lambda x: max(0, x), x_range=[-4, 4], color=m.RED)
        softplus = axes.plot(lambda x: np.log(1 + np.exp(x)), x_range=[-4, 4], color=m.BLUE)
        identity = axes.plot(lambda x: x, x_range=[-4, 4], color=m.GREEN, stroke_width=1)

        label_relu = m.MathTex("\\text{ReLU}(x)", color=m.RED).next_to(relu.get_end(), m.UR)
        label_softplus = m.MathTex("\\text{Softplus}(x)", color=m.BLUE).next_to(softplus.get_end(), m.UR)
        label_identity = m.MathTex("y=x", color=m.GREEN).next_to(identity.get_end(), m.UR)

        self.add(relu, softplus, identity, label_relu, label_softplus, label_identity)

        caption = m.Text(
            "Softplus — гладкая аппроксимация ReLU",
            font_size=24,
            color=m.YELLOW
        ).to_edge(m.DOWN)
        self.add(caption)
        self.wait(2)


# ============================================================
# 6. Приближения Тейлора для e^x
# ============================================================
class TaylorExp(Scene_):
    """
    Последовательно добавляются полиномы Тейлора возрастающего порядка.
    """
    def construct(self):
        axes = m.Axes(
            x_range=[-2, 2, 1],
            y_range=[-1, 8, 2],
            x_length=8,
            y_length=6,
            axis_config={"color": m.GRAY}
        )
        axes.add_coordinates()
        self.add(axes)

        self.play(m.Create(exp_graph := axes.plot(lambda x: np.exp(x), x_range=[-2, 2], color=m.WHITE)))
        label_exp = m.MathTex("e^x", color=m.WHITE).next_to(exp_graph.get_end(), m.UL)
        self.add(exp_graph, label_exp)
        self.wait(1)

        def taylor(x, n):
            res = 0
            fact = 1
            for i in range(n + 1):
                if i > 0:
                    fact *= i
                res += x**i / fact
            return res

        colors = [m.BLUE, m.GREEN, m.YELLOW, m.ORANGE, m.PURPLE, m.PINK]
        labels = ["n=0", "n=1", "n=2", "n=3", "n=4", "n=5"]

        for n in range(6):
            graph = axes.plot(lambda x: taylor(x, n), x_range=[-2, 2], color=colors[n], stroke_width=2)
            label = m.MathTex(labels[n], color=colors[n]).next_to(graph.get_end(), m.UR)
            self.play(m.Create(graph), m.Write(label), run_time=0.5)
            self.wait(0.5)
            # self.remove(graph)

        caption = m.Text(
            "Приближения Тейлора для e^x",
            font_size=24,
            color=m.YELLOW
        ).to_edge(m.DOWN)
        self.add(caption)
        self.wait(2)


# ============================================================
# 7. Метод Ньютона (квадратичное приближение)
# ============================================================
class NewtonMethod(Scene_):
    """
    Строится парабола Тейлора, её вершина — следующая точка.
    """
    def construct(self):
        axes = m.Axes(
            x_range=[-3, 5, 1],
            y_range=[-2, 10, 2],
            x_length=8,
            y_length=6,
            axis_config={"color": m.GRAY}
        )
        axes.add_coordinates()
        self.add(axes)

        L = lambda x: x**4 - 3 * x**2 + x + 2
        dL = lambda x: 4 * x**3 - 6 * x + 1
        ddL = lambda x: 12 * x**2 - 6

        graph = axes.plot(L, x_range=[-2.5, 3.5], color=m.BLUE)
        label = m.MathTex("L(x)", color=m.BLUE).next_to(graph.get_end(), m.UR)
        self.add(graph, label)

        x = m.ValueTracker(-2)
        dot = m.Dot(axes.coords_to_point(x.get_value(), L(x.get_value())), color=m.RED)
        self.add(dot)

        def taylor_parabola(x0):
            return lambda x: L(x0) + dL(x0) * (x - x0) + 0.5 * ddL(x0) * (x - x0) ** 2

        for _ in range(4):
            x0 = x.get_value()
            y0 = L(x0)
            dy = dL(x0)
            ddy = ddL(x0)
            if abs(ddy) < 1e-6:
                break
            x1 = x0 - dy / ddy

            parab = axes.plot(taylor_parabola(x0), x_range=[x0 - 2, x0 + 2], color=m.YELLOW, stroke_width=2)
            self.play(m.Create(parab), run_time=0.5)

            dot1 = m.Dot(axes.coords_to_point(x1, L(x1)), color=m.GREEN)
            self.play(m.Create(dot1), run_time=0.5)

            self.play(
                x.animate.set_value(x1),
                dot.animate.move_to(axes.coords_to_point(x1, L(x1))),
                run_time=0.5
            )
            self.wait(0.2)

        caption = m.Text(
            "Метод Ньютона: парабола → вершина → новая точка",
            font_size=24,
            color=m.YELLOW
        ).to_edge(m.DOWN)
        self.add(caption)
        self.wait(2)


# ============================================================
# 8. Частные производные на 3D-поверхности
# ============================================================
class PartialDerivatives3D(ThreeDScene_):
    """
    Поверхность z = x^2 + y^2, две кривые и касательные к ним.
    """
    def construct(self):
        # ---------- камера ----------
        self.set_camera_orientation(
            phi=47 * m.DEGREES,
            theta=-45 * m.DEGREES,
            zoom=0.95,
        )

        # ---------- оси ----------
        axes = m.ThreeDAxes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            z_range=[-0.5, 3.5, 1],
            x_length=6.5,
            y_length=6.5,
            z_length=4.0,
            axis_config={"color": m.GRAY, "stroke_width": 2},
        )
        self.add(axes)

        # Подписи осей (всегда лицом к камере)
        x_lbl = m.MathTex("x", font_size=30).move_to(axes.c2p(2.35, 0, 0))
        y_lbl = m.MathTex("y", font_size=30).move_to(axes.c2p(0, 2.35, 0))
        z_lbl = m.MathTex("z", font_size=30).move_to(axes.c2p(0, 0, 3.8))
        self.add_fixed_orientation_mobjects(x_lbl, y_lbl, z_lbl)
        self.add(x_lbl, y_lbl, z_lbl)

        # ---------- функция и выделенная точка ----------
        f = lambda x, y: 0.5 * x ** 2 + 1.2 * y ** 2
        x0, y0 = 0.9, 0.9
        z0 = f(x0, y0)

        # ---------- поверхность ----------
        surface = m.Surface(
            lambda u, v: axes.c2p(u, v, f(u, v)),
            u_range=[-1.6, 1.6],
            v_range=[-1.6, 1.6],
            resolution=(40, 40),
            fill_opacity=0.45,
            checkerboard_colors=[m.BLUE_E, m.BLUE_D],
        )
        self.add(surface)

        # ---------- полупрозрачные секущие плоскости ----------
        # Плоскость y = y0 (красная) — параллельна плоскости xz
        red_plane = m.Polygon(
            axes.c2p(-1.7, y0, 0),
            axes.c2p(1.7, y0, 0),
            axes.c2p(1.7, y0, 4.0),
            axes.c2p(-1.7, y0, 4.0),
            color=m.RED,
            fill_opacity=0.07,
            stroke_width=0,
        )
        # Плоскость x = x0 (синяя) — параллельна плоскости yz
        blue_plane = m.Polygon(
            axes.c2p(x0, -1.7, 0),
            axes.c2p(x0, 1.7, 0),
            axes.c2p(x0, 1.7, 4.0),
            axes.c2p(x0, -1.7, 4.0),
            color=m.BLUE,
            fill_opacity=0.07,
            stroke_width=0,
        )
        self.add(red_plane, blue_plane)

        # ---------- кривые на поверхности ----------
        # Красная: y = y0, параметр t = x
        red_curve = m.ParametricFunction(
            lambda t: axes.c2p(t, y0, f(t, y0)),
            t_range=[-1.6, 1.6],
            color=m.RED,
        )
        red_curve.set_stroke(width=5)

        # Синяя: x = x0, параметр t = y
        blue_curve = m.ParametricFunction(
            lambda t: axes.c2p(x0, t, f(x0, t)),
            t_range=[-1.6, 1.6],
            color=m.BLUE,
        )
        blue_curve.set_stroke(width=5)

        self.add(red_curve, blue_curve)

        # ---------- точка пересечения ----------
        P = axes.c2p(x0, y0, z0)
        point = m.Dot3D(P, color=m.YELLOW, radius=0.09)
        self.add(point)

        # ---------- касательные ----------
        # ∂f/∂x = x  ⇒  в точке x0 = 0.9  наклон = 0.9
        # ∂f/∂y = 2.4y ⇒  в точке y0 = 0.9  наклон = 2.16
        dfdx = x0
        dfdy = 2.4 * y0

        Lr = 0.85
        tangent_red = m.Line3D(
            axes.c2p(x0 - Lr, y0, z0 - Lr * dfdx),
            axes.c2p(x0 + Lr, y0, z0 + Lr * dfdx),
            color=m.RED,
            thickness=0.022,
        )
        Lb = 0.55
        tangent_blue = m.Line3D(
            axes.c2p(x0, y0 - Lb, z0 - Lb * dfdy),
            axes.c2p(x0, y0 + Lb, z0 + Lb * dfdy),
            color=m.BLUE,
            thickness=0.022,
        )
        self.add(tangent_red, tangent_blue)

        # ---------- вертикальный пунктир до плоскости z = 0 ----------
        drop = m.DashedLine(
            P,
            axes.c2p(x0, y0, 0),
            color=m.YELLOW,
            stroke_width=2,
            dash_length=0.08,
        )
        self.add(drop)

        # ---------- 3D-подписи у кривых (лицом к камере) ----------
        red_lbl_3d = m.MathTex(
            r"y = y_0", color=m.RED, font_size=26
        ).move_to(axes.c2p(-1.4, y0, f(-1.4, y0) + 0.35))
        blue_lbl_3d = m.MathTex(
            r"x = x_0", color=m.BLUE, font_size=26
        ).move_to(axes.c2p(x0, -1.4, f(x0, -1.4) + 0.35))
        self.add_fixed_orientation_mobjects(red_lbl_3d, blue_lbl_3d)
        self.add(red_lbl_3d, blue_lbl_3d)

        # ---------- подписи, фиксированные в кадре ----------
        title = m.Text(
            "Частные производные",
            font_size=28,
            color=m.WHITE,
        ).to_edge(m.UP, buff=0.3)

        # Легенда
        red_swatch = m.Square(side_length=0.3, color=m.RED, fill_opacity=1)
        red_text = m.Text(
            "y = y0:  наклон = ∂f/∂x", font_size=20, color=m.RED
        )
        red_row = m.VGroup(red_swatch, red_text).arrange(m.RIGHT, buff=0.15)

        blue_swatch = m.Square(side_length=0.3, color=m.BLUE, fill_opacity=1)
        blue_text = m.Text(
            "x = x0:  наклон = ∂f/∂y", font_size=20, color=m.BLUE
        )
        blue_row = m.VGroup(blue_swatch, blue_text).arrange(m.RIGHT, buff=0.15)

        legend = m.VGroup(red_row, blue_row).arrange(
            m.DOWN, buff=0.15, aligned_edge=m.LEFT
        )
        legend.to_corner(m.UL, buff=0.35)

        # Основная подпись
        caption = m.Text(
            "Наклон красной касательной — частная производная по x, синей — по y",
            font_size=20,
            color=m.YELLOW,
        ).to_edge(m.DOWN, buff=0.35)

        self.add_fixed_in_frame_mobjects(title, legend, caption)
        self.add(title, legend, caption)

        self.wait(2)


# ============================================================
# 9. Карта уровней: градиентный спуск vs метод Ньютона
# ============================================================
class LevelSetsPaths(Scene_):
    """
    Эллиптические линии уровня, два пути к минимуму.
    """
    def construct(self):
        axes = m.Axes(
            x_range=[-5, 5, 1],
            y_range=[-3, 3, 1],
            x_length=8,
            y_length=8,
            axis_config={"color": m.GRAY}
        )
        self.add(axes)

        # Линии уровня L(x,y) = x^2 + 4y^2
        for level in [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25]:
            ellipse = m.Ellipse(
                width=2 * np.sqrt(level),
                height=np.sqrt(level),
                color=m.GRAY,
                stroke_opacity=0.5
            )
            ellipse.move_to(axes.coords_to_point(0, 0))
            self.add(ellipse)

        min_point = m.Dot(axes.coords_to_point(0, 0), color=m.RED)
        self.add(min_point)

        start = np.array([2.0, 1.5])
        start_dot = m.Dot(axes.coords_to_point(*start), color=m.GREEN)
        self.add(start_dot)

        # Градиентный спуск
        lr = 0.1
        path_gd = [start]
        x, y = start
        for _ in range(20):
            grad = np.array([2 * x, 8 * y])
            x, y = x - lr * grad[0], y - lr * grad[1]
            path_gd.append(np.array([x, y]))

        gd_line = m.VMobject()
        gd_line.set_points_as_corners([axes.coords_to_point(*p) for p in path_gd])
        gd_line.set_color(m.BLUE)
        self.add(gd_line)

        # Метод Ньютона — сразу к минимуму (квадратичная функция)
        newton_line = m.Line(
            axes.coords_to_point(*start),
            axes.coords_to_point(0, 0),
            color=m.YELLOW
        )
        self.add(newton_line)

        label_gd = m.Text("Градиентный спуск", color=m.BLUE).scale(0.6).shift(m.UP*0.5 + m.RIGHT*3.5)
        label_newton = m.Text("Метод Ньютона", color=m.YELLOW).scale(0.6).shift(m.UP*1.5 + m.LEFT)
        self.add(label_gd, label_newton)

        # caption = m.Text(
        #     "Градиентный спуск зигзагом, Ньютон — напрямую",
        #     font_size=24,
        #     color=m.YELLOW
        # ).to_edge(m.DOWN)
        # self.add(caption)
        self.wait(2)


if __name__ == '__main__':
    import os
    from pathlib import Path

    SCENES = [
        # "SecantToTangent",
        # "ExtremaDerivative",
        # "ZoomToTangent",
        # "GradientDescentParabola",
        # "ReLU_Softplus",
        # "TaylorExp",
        # "NewtonMethod",
        # "PartialDerivatives3D",
        "LevelSetsPaths",
    ]
    file_path = Path(__file__).resolve()

    for SCENE in SCENES:
        os.system(f"manim {file_path} {SCENE} -qh")
        os.system(f"manim {file_path} {SCENE} -sqh")