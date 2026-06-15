# Extended Big Five questionnaire with subtraits and their questions
questionnaire = {
    "openness": {
        "imagination": [
            "I enjoy daydreaming or thinking about abstract, fantastical ideas.",
            "When solving problems, I often come up with creative or unconventional solutions."
        ],
        "aesthetic_sensitivity": [
            "I am deeply moved by art, music, or nature.",
            "I often seek beauty in my surroundings, such as enjoying sunsets or well-designed spaces."
        ],
        "intellectual_curiosity": [
            "I enjoy learning about new topics just for the sake of knowledge.",
            "I am drawn to complex or theoretical ideas, like philosophy or quantum mechanics."
        ],
        "adventure_seeking": [
            "I prefer trying new activities, like traveling to unfamiliar places or sampling exotic foods.",
            "I am comfortable taking risks to explore new opportunities."
        ],
        "emotional_openness": [
            "I am willing to express my deeper feelings, even if they’re complicated or vulnerable.",
            "I often reflect on how my emotions shape my experiences."
        ],
    },
    "conscientiousness": {
        "self_discipline": [
            "I can persist with tasks even when they become boring or difficult.",
            "I often complete projects ahead of deadlines."
        ],
        "orderliness": [
            "I like to keep my workspace, home, or schedule well-organized.",
            "I feel uncomfortable in cluttered or chaotic environments."
        ],
        "dutifulness": [
            "I feel a strong obligation to fulfill my commitments, even when it’s inconvenient.",
            "I often feel guilty if I don’t meet others’ expectations."
        ],
        "achievement_striving": [
            "I set ambitious goals for myself and work hard to achieve them.",
            "I enjoy feeling productive and accomplished after a busy day."
        ],
        "cautiousness": [
            "I take time to weigh the pros and cons before making decisions.",
            "I am careful to avoid risks, even if they might lead to big rewards."
        ],
    },
    "extraversion": {
        "sociability": [
            "I feel energized after spending time with others.",
            "I enjoy large gatherings and meeting new people."
        ],
        "assertiveness": [
            "I am confident in expressing my opinions, even in group settings.",
            "I often take the lead in organizing events or activities."
        ],
        "energy_level": [
            "I have a lot of physical and mental energy throughout the day.",
            "I enjoy fast-paced environments with constant stimulation."
        ],
        "excitement_seeking": [
            "I am drawn to thrilling activities, such as roller coasters, skydiving, or adventurous travel.",
            "I get bored quickly in routine or low-energy settings."
        ],
        "positive_emotions": [
            "I often feel cheerful, enthusiastic, and optimistic.",
            "I am good at lifting the mood of people around me."
        ],
    },
    "agreeableness": {
        "trust": [
            "I believe most people have good intentions.",
            "I am comfortable relying on others to do their part in a group project."
        ],
        "altruism": [
            "I enjoy helping others, even if it requires extra effort or sacrifice.",
            "I find satisfaction in volunteering or supporting a cause."
        ],
        "modesty": [
            "I feel uncomfortable boasting about my achievements or skills.",
            "I avoid drawing attention to myself, even when I deserve recognition."
        ],
        "compassion": [
            "I am quick to notice when others are upset or in need of comfort.",
            "I go out of my way to make others feel cared for and supported."
        ],
        "cooperation": [
            "I am willing to compromise to avoid conflict.",
            "I prioritize group harmony over my own preferences in team settings."
        ],
    },
    "neuroticism": {
        "anxiety": [
            "I often worry about future events or possible problems.",
            "I feel tense or nervous in unfamiliar or high-pressure situations."
        ],
        "anger": [
            "I feel frustrated or irritated easily.",
            "Small annoyances sometimes make me lose my temper."
        ],
        "depression": [
            "I often feel sad, discouraged, or unmotivated, even when there’s no clear reason.",
            "I find it hard to enjoy activities that used to make me happy."
        ],
        "self_consciousness": [
            "I am overly concerned about what others think of me.",
            "I often feel embarrassed or judged in social situations."
        ],
        "vulnerability": [
            "I find it difficult to cope with stressful situations or major life changes.",
            "I feel overwhelmed when dealing with challenges, even if they’re manageable."
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