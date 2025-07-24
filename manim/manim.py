# %%manim -qh Detector
from manim import *
import random
import numpy as np


class Detector(Scene):
    def construct(self):
        # --- Parameters ---
        T_START_DRIFT = 10
        T_END_DRIFT = 15
        T_TOTAL = 25

        STREAM_LENGTH = 1000
        VISIBLE_WINDOW_DURATION = 5.0
        STREAM_X_LEFT = -7
        STREAM_X_RIGHT = -2.25
  

        time = ValueTracker(0)
        title = Text("Example of Concept Drift Detection", font_size=32).to_edge(DOWN)

        # --- Stream  ---
        def generate_weighted_dice(inverse=True):
            if not inverse:
                return random.choices(
                    population=[1, 2, 3, 4, 5, 6],
                    weights=[30, 15, 5, 5, 15, 30],
                    k=1
                )[0]
            else:
                return random.choices(
                    population=[1, 2, 3, 4, 5, 6],
                    weights=[5, 15, 30, 30, 15, 5],
                    k=1
                )[0]

        def get_stream_value(i, t):
            if t < T_START_DRIFT:
                return generate_weighted_dice(inverse=True)
            elif t > T_END_DRIFT:
                return generate_weighted_dice()
            else:
                alpha = (t - T_START_DRIFT) / (T_END_DRIFT - T_START_DRIFT)
                v1 = generate_weighted_dice(inverse=True)
                v2 = generate_weighted_dice()
                return np.interp(alpha, [0, 1], [v1, v2])

        stream_times = np.linspace(0, T_TOTAL, STREAM_LENGTH)
        stream_values = [get_stream_value(0, t) for t in stream_times]

        def get_visible_stream_points(current_time):
            window_size = int(STREAM_LENGTH * VISIBLE_WINDOW_DURATION / T_TOTAL)
            end_idx = np.searchsorted(stream_times, current_time)
            start_idx = end_idx - window_size
            
            if start_idx < 0:
                # Create flowing data even before we reach actual stream data
                y_window = [(stream_values[i] - 3.5) * 0.4 for i in range(0, end_idx)]
                remaining = -start_idx
                if remaining > 0:
                    # Generate flowing values that move with time
                    flow_offset = current_time * STREAM_LENGTH / T_TOTAL
                    extended_values = []
                    for i in range(remaining):
                        # Use actual stream values but offset by flow to create movement
                        idx = int((i + flow_offset) % len(stream_values))
                        extended_values.append((stream_values[idx] - 3.5) * 0.4)
                    y_screen = extended_values + y_window
                else:
                    y_screen = y_window
            else:
                y_screen = [(stream_values[i] - 3.5) * 0.4 for i in range(start_idx, end_idx)]
            
            y_screen = list(reversed(y_screen))
            x_screen = np.linspace(STREAM_X_LEFT, STREAM_X_RIGHT, len(y_screen))
            points = [[x, y, 0] for x, y in zip(x_screen, y_screen)]
            if len(points) < 2:
                y = (stream_values[0] - 3.5) * 0.4
                points = [[STREAM_X_LEFT, y, 0], [STREAM_X_RIGHT, y, 0]]
            return points

        stream = always_redraw(lambda:
            VMobject().set_points_smoothly(get_visible_stream_points(time.get_value()))
                .set_color(BLUE).set_stroke(width=2)
        )

        # --- Classifier Box ---
        box = Rectangle(width=4.5, height=2.5, color=PURPLE, fill_color=GRAY, fill_opacity=0.3).move_to(ORIGIN)
        classifier_text = Text("Classifier", font_size=28).move_to(box.get_center())
        detector_label = Text("Detector", font_size=20).next_to(box, UP, buff=-0.3)

        # --- Explanation Text ---
        explanation_1 = Text("Classifier trains from stream", font_size=27).to_corner(UR, buff=0.5)
        explanation_2 = Text("Performance degrades after drift", font_size=27).to_corner(UR, buff=0.5)
        explanation_3 = Text("Detector recognizes drift, accuracy recovers", font_size=27).to_corner(UR, buff=0.5)
        
        current_explanation = explanation_1.copy()

        # --- Accuracy Graph ---
        graph_width = 3.0
        graph_height = 2.0
        accuracy_x = np.linspace(0, T_TOTAL, 40)
        graph_origin = box.get_right() + RIGHT * 1.2 + DOWN * graph_height/2
        graph_background = Rectangle(
            width=graph_width, 
            height=graph_height, 
            color=WHITE, 
            fill_opacity=0.1
        ).move_to(graph_origin + RIGHT * graph_width/2 + UP * graph_height/2)
        x_axis = Line(
            start=graph_origin,
            end=graph_origin + RIGHT * graph_width,
            color=WHITE
        )
        y_axis = Line(
            start=graph_origin,
            end=graph_origin + UP * graph_height,
            color=WHITE
        )
        x_label = Text("Time", font_size=16).next_to(x_axis, DOWN, buff=0.1)
        y_label = Text("Accuracy", font_size=16).next_to(y_axis, LEFT, buff=-0.25).rotate(PI/2)
        y_tick_0 = Text("0%", font_size=12).next_to(graph_origin, LEFT, buff=0.1)
        y_tick_1 = Text("100%", font_size=12).next_to(graph_origin + UP * graph_height, LEFT, buff=0.1)

        accuracy_curve = []
        for x in accuracy_x:
            if x < T_START_DRIFT:
                scale = T_START_DRIFT / 3
                acc = 0.9 * (1 - np.exp(-x / scale))
            elif x < T_END_DRIFT:
                acc = 0.9 - 0.7 * ((x - T_START_DRIFT) / (T_END_DRIFT - T_START_DRIFT))
            elif x < T_END_DRIFT + 5:
                recovery_time = x - T_END_DRIFT
                scale = 5 / 3
                recovery_amount = 0.7 * (1 - np.exp(-recovery_time / scale))
                acc = 0.2 + recovery_amount
            else:
                acc = 0.9
            noise = np.random.uniform(-0.03, 0.03)
            acc = np.clip(acc + noise, 0, 1)
            accuracy_curve.append(acc)

        def get_accuracy_curve(t):
            shown_indices = accuracy_x <= t
            shown_x = accuracy_x[shown_indices]
            shown_y = [accuracy_curve[i] for i in range(len(shown_x))]
            return shown_x, shown_y

        def get_graph_points(shown_x, shown_y):
            if len(shown_x) <= 1:
                return [graph_origin, graph_origin]
            x_min, x_max = 0, T_TOTAL
            y_min, y_max = 0, 1
            return [
                graph_origin
                + RIGHT * (graph_width * (x - x_min) / (x_max - x_min))
                + UP * (graph_height * (y - y_min) / (y_max - y_min))
                for x, y in zip(shown_x, shown_y)
            ]

        accuracy_graph = always_redraw(lambda: (
            lambda shown_x, shown_y:
                VMobject().set_points_as_corners(get_graph_points(shown_x, shown_y)).set_color(BLUE).set_stroke(width=3)
            )(*get_accuracy_curve(time.get_value()))
        )

        # --- Alert ---
        siren = ArcBetweenPoints(
            start=box.get_top()-0.1 + LEFT * 0.3,
            end=box.get_top()-0.1 + RIGHT * 0.3,
            angle=-PI,
            color=RED
        ).shift(UP * 0.15)
        red_flash = ArcBetweenPoints(
            start=box.get_top()-0.1 + LEFT * 0.5,
            end=box.get_top()-0.1 + RIGHT * 0.5,
            angle=-PI,
            color=RED
        ).shift(UP * 0.15)
        red_flash.set_opacity(0)
        
        change_happening_alert = Text("⚠ Change Happening", color=RED, font_size=28).next_to(stream, UP * 5)
        change_detected_alert = Text("✔ Change Detected", color=RED, font_size=28).next_to(box, UP * 2.5)
        current_change_alert = None
        current_detection_alert = None

        # --- Animation Sequence ---

        # Phase 0: Startup
        self.play(Write(title))

        # Classifier
        self.play(Create(box), FadeIn(classifier_text), Write(detector_label), FadeIn(siren))

        # Graph
        self.play(
            Create(graph_background),
            Create(x_axis),
            Create(y_axis),
            Write(x_label),
            Write(y_label),
            Write(y_tick_0),
            Write(y_tick_1)
        )
        self.play(FadeIn(accuracy_graph))
        
        self.play(Create(stream))
        self.add(red_flash)
        self.play(FadeIn(current_explanation))
        

        # Phase 1: Training phase
        self.play(
            time.animate.set_value(T_START_DRIFT), 
            run_time=T_START_DRIFT,
            rate_func=linear
        )

        # Transition 1->2
        self.play(Transform(current_explanation, explanation_2), run_time=0.5)
        current_change_alert = change_happening_alert.copy()
        self.play(FadeIn(current_change_alert))
        
        # Phase 2: Drift phase
        self.play(
            time.animate.set_value(T_END_DRIFT), 
            run_time=(T_END_DRIFT - T_START_DRIFT),
            rate_func=linear
        )
        
        # Transition 2->3
        self.play(FadeOut(current_change_alert))
        current_detection_alert = change_detected_alert.copy()
        self.play(FadeIn(current_detection_alert), red_flash.animate.set_opacity(0.2))
        self.play(Transform(current_explanation, explanation_3), run_time=0.5)
        
        # Phase 3: Recovery phase
        self.play(
            time.animate.set_value(T_TOTAL), 
            run_time=(T_TOTAL - T_END_DRIFT),
            rate_func=linear
        )

        
