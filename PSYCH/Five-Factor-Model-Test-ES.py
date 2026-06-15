# Extended Big Five questionnaire with subtraits and their questions
questionnaire = {
    "openness": {
        "imagination": [
            "Me gusta soñar despierto o pensar en ideas abstractas y fantásticas.",
            "Al resolver problemas, a menudo se me ocurren soluciones creativas o poco convencionales."
        ],
        "aesthetic_sensitivity": [
            "Me conmueve profundamente el arte, la música o la naturaleza.",
            "A menudo busco la belleza en mi entorno, como disfrutar de las puestas de sol o de espacios bien diseñados."
        ],
        "intellectual_curiosity": [
            "Me gusta aprender sobre nuevos temas solo por el placer de aprender.",
            "Me atraen las ideas complejas o teóricas, como la filosofía o la mecánica cuántica."
        ],
        "adventure_seeking": [
            "Prefiero probar nuevas actividades, como viajar a lugares desconocidos o probar comidas exóticas.",
            "Me siento cómodo tomando riesgos para explorar nuevas oportunidades."
        ],
        "emotional_openness": [
            "Estoy dispuesto a expresar mis sentimientos más profundos, incluso si son complicados o vulnerables.",
            "A menudo reflexiono sobre cómo mis emociones dan forma a mis experiencias."
        ],
    },
    "conscientiousness": {
        "self_discipline": [
            "Puedo persistir con las tareas incluso cuando se vuelven aburridas o difíciles.",
            "A menudo completo proyectos antes de las fechas límite."
        ],
        "orderliness": [
            "Me gusta mantener mi espacio de trabajo, mi hogar o mi horario bien organizado.",
            "Me siento incómodo en entornos desordenados o caóticos."
        ],
        "dutifulness": [
            "Siento una fuerte obligación de cumplir con mis compromisos, incluso cuando es inconveniente.",
            "A menudo me siento culpable si no cumplo con las expectativas de los demás."
        ],
        "achievement_striving": [
            "Me fijo metas ambiciosas y trabajo duro para alcanzarlas.",
            "Disfruto de sentirme productivo y realizado después de un día ajetreado."
        ],
        "cautiousness": [
            "Me tomo el tiempo de sopesar los pros y los contras antes de tomar decisiones.",
            "Tengo cuidado de evitar riesgos, incluso si pueden dar lugar a grandes recompensas."
        ],
    },
    "extraversion": {
        "sociability": [
            "Me siento con energía después de pasar tiempo con otras personas.",
            "Disfruto de las reuniones grandes y de conocer gente nueva."
        ],
        "assertiveness": [
            "Tengo confianza para expresar mis opiniones, incluso en entornos grupales.",
            "A menudo tomo la iniciativa en la organización de eventos o actividades."
        ],
        "energy_level": [
            "Tengo mucha energía física y mental durante todo el día.",
            "Disfruto de los entornos de ritmo rápido con estimulación constante."
        ],
        "excitement_seeking": [
            "Me atraen las actividades emocionantes, como las montañas rusas, el paracaidismo o los viajes de aventura.",
            "Me aburro rápidamente en entornos rutinarios o de baja energía."
        ],
        "positive_emotions": [
            "A menudo me siento alegre, entusiasta y optimista.",
            "Soy bueno para levantar el ánimo de las personas que me rodean."
        ],
    },
    "agreeableness": {
        "trust": [
            "Creo que la mayoría de las personas tienen buenas intenciones.",
            "Me siento cómodo confiando en que los demás hagan su parte en un proyecto grupal."
        ],
        "altruism": [
            "Disfruto ayudando a los demás, incluso si requiere un esfuerzo o sacrificio adicional.",
            "Encuentro satisfacción en el voluntariado o en apoyar una causa."
        ],
        "modesty": [
            "Me siento incómodo alardeando de mis logros o habilidades.",
            "Evito llamar la atención sobre mí mismo, incluso cuando merezco reconocimiento."
        ],
        "compassion": [
            "Me doy cuenta rápidamente cuando los demás están molestos o necesitan consuelo.",
            "Hago todo lo posible para que los demás se sientan cuidados y apoyados."
        ],
        "cooperation": [
            "Estoy dispuesto a hacer concesiones para evitar conflictos.",
            "Priorizo la armonía del grupo por encima de mis propias preferencias en entornos de equipo."
        ],
    },
    "neuroticism": {
        "anxiety": [
            "A menudo me preocupan los acontecimientos futuros o los posibles problemas.",
            "Me siento tenso o nervioso en situaciones desconocidas o de mucha presión."
        ],
        "anger": [
            "Me siento frustrado o irritado fácilmente.",
            "A veces, las pequeñas molestias me hacen perder los estribos."
        ],
        "depression": [
            "A menudo me siento triste, desanimado o desmotivado, incluso cuando no hay una razón clara.",
            "Me resulta difícil disfrutar de actividades que antes me hacían feliz."
        ],
        "self_consciousness": [
            "Me preocupa demasiado lo que los demás piensen de mí.",
            "A menudo me siento avergonzado o juzgado en situaciones sociales."
        ],
        "vulnerability": [
            "Me resulta difícil afrontar situaciones estresantes o grandes cambios en la vida.",
            "Me siento abrumado cuando me enfrento a desafíos, incluso si son manejables."
        ],
    },
}

import json
import numpy as np
import gradio as gr
import plotly.graph_objects as go
from scipy.stats import percentileofscore

# Define TRAIT_COLORS
TRAIT_COLORS = {
    "openness": "blue",
    "conscientiousness": "green",
    "extraversion": "orange",
    "agreeableness": "purple",
    "neuroticism": "red"
}

# Flatten questions dynamically
def build_questions():
    return [
        (trait, sub_trait, q)
        for trait, sub_traits in questionnaire.items()
        for sub_trait, qs in sub_traits.items()
        for q in qs
    ]

questions = build_questions()

# Initialize state
state = {"current_question": 0, "responses": []}

# Compute scores with percentiles and z-scores
def compute_scores_and_percentiles(responses):
    scores = {}
    idx = 0
    for trait, sub_traits in questionnaire.items():
        for sub_trait, qs in sub_traits.items():
            mean_score = np.mean(responses[idx:idx + len(qs)])
            scores[f"{trait}_{sub_trait}"] = mean_score
            idx += len(qs)
    
    # Convert scores to arrays for percentile/z-score calculations
    values = np.array(list(scores.values()))
    z_scores = (values - np.mean(values)) / np.std(values)
    percentiles = [percentileofscore(values, score) for score in values]
    
    return scores, z_scores, percentiles

# Create chart with colors, percentiles, and z-scores
def create_chart(scores, z_scores, percentiles):
    subtraits = [key.split("_")[1] for key in scores.keys()]
    values = list(scores.values())
    trait_keys = [key.split("_")[0] for key in scores.keys()]
    colors = [TRAIT_COLORS[trait] for trait in trait_keys]

    # Create a bar chart
    fig = go.Figure()

    for i, (trait, color) in enumerate(TRAIT_COLORS.items()):
        indices = [j for j, t in enumerate(trait_keys) if t == trait]
        trait_subtraits = [subtraits[j] for j in indices]
        trait_values = [values[j] for j in indices]
        trait_z_scores = [z_scores[j] for j in indices]
        trait_percentiles = [percentiles[j] for j in indices]

        fig.add_trace(
            go.Bar(
                x=trait_subtraits,
                y=trait_values,
                name=trait.capitalize(),
                marker_color=color,
                text=[
                    f"Score: {v:.2f}<br>Z-score: {z:.2f}<br>Percentile: {p:.1f}%"
                    for v, z, p in zip(trait_values, trait_z_scores, trait_percentiles)
                ],
                hoverinfo="text"
            )
        )

    fig.update_layout(
        title="Trait Breakdown with Percentiles and Z-Scores",
        xaxis_title="Subtraits",
        yaxis_title="Average Score (1-10)",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        legend=dict(
            title="Traits",
            bgcolor="black",
            bordercolor="gray",
            borderwidth=1
        )
    )
    return fig

# Progress gauge
def plot_progress(current, total, question_text, question_num):
    progress = (current / total) * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=progress,
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': 'purple'},
            'bgcolor': 'black',
            'steps': [
                {'range': [0, 25], 'color': 'darkviolet'},
                {'range': [25, 50], 'color': 'violet'},
                {'range': [50, 75], 'color': 'magenta'},
                {'range': [75, 100], 'color': 'plum'}
            ],
        },
        domain={'x': [0, 1], 'y': [0, 1]}
    ))

    fig.add_annotation(
        x=0.5, y=-0.2, text=f"Question {question_num} / {total}", showarrow=False,
        font=dict(size=14, color='white'), align="center"
    )

    fig.update_layout(
        title={
            'text': f"<b>{question_text}</b>",
            'font': {'size': 20, 'color': "white"},
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.85,
        },
        margin=dict(t=170),
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
    )

    return fig

# Start test
def start_test():
    state["current_question"] = 0
    state["responses"] = []
    question = questions[state["current_question"]][2]
    return (
        question,
        plot_progress(0, len(questions), question, 1),
        gr.update(visible=True),
        gr.update(visible=False),
    )





# Save Plotly chart as HTML
def save_plotly_html(chart, file_name="results_chart.html"):
    chart.write_html(file_name)
    print(f"Chart saved to {file_name}")

# Save test results as a JSON file
def save_results(responses, file_name="test_results.json"):
    with open(file_name, "w") as f:
        json.dump({"responses": responses}, f, indent=4)
    print(f"Results saved to {file_name}")

# Modified next_question to save results and chart
def next_question(response):
    state["responses"].append(int(response))
    state["current_question"] += 1

    if state["current_question"] >= len(questions):
        scores, z_scores, percentiles = compute_scores_and_percentiles(state["responses"])
        result_chart = create_chart(scores, z_scores, percentiles)

        # Save results and chart
        save_results(state["responses"], "test_results.json")
        save_plotly_html(result_chart, "results_chart.html")

        return "Test Complete! Your results:", result_chart, gr.update(visible=False), gr.update(visible=True)

    question = questions[state["current_question"]][2]
    return (
        question,
        plot_progress(
            state["current_question"], len(questions),
            question, state["current_question"] + 1
        ),
        gr.update(visible=True),
        gr.update(visible=False),
    )

def create_gradio_app():
    with gr.Blocks() as app:
        gr.Markdown("## Extended Big Five Personality Test")

        # UI Elements
        start_button = gr.Button("Start Test")
        question_text = gr.Textbox(label="Question", interactive=False, visible=True)
        button_group = gr.Radio([str(i) for i in range(1, 11)], label="Your Response (1-10)", visible=True)
        progress_gauge = gr.Plot()
        result_output = gr.Textbox(label="Results", visible=True)
        download_results = gr.File(label="Download Results JSON", visible=False)
        download_chart = gr.File(label="Download Chart HTML", visible=True)

        # Start Test Logic
        start_button.click(
            start_test,
            outputs=[question_text, progress_gauge, button_group, result_output]
        )

        # Question Logic
        def handle_next_question(response):
            output = next_question(response)

            # Check if the test is complete to enable downloads
            if state["current_question"] >= len(questions):
                return (*output, "test_results.json", "results_chart.html")
            else:
                return (*output, None, None)

        # Respond to button changes and enable downloads after test completion
        button_group.change(
            handle_next_question,
            inputs=button_group,
            outputs=[
                question_text, progress_gauge, button_group, result_output,
                download_results, download_chart
            ]
        )

    return app


# Launch the Gradio app
app = create_gradio_app()

# Launch the app locally
app.launch()