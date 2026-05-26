from google import genai

_client = None

def configure_gemini(api_key):
    global _client
    _client = genai.Client(api_key=api_key)

def get_ai_analysis(player_name, scores, weaknesses, strategies):
    try:
        weakness_text = ', '.join(weaknesses) if weaknesses else 'None detected'
        top_strategy = strategies[0] if strategies else {}
        strategy_text = f"Line: {top_strategy.get('line','N/A')}, Length: {top_strategy.get('length','N/A')}, Speed: {top_strategy.get('speed','N/A')}, Variation: {top_strategy.get('variation','N/A')}" if top_strategy else "Standard line and length"

        prompt = f"""You are an expert cricket batting coach with 20+ years of experience.

Analyze this batsman's performance and give personalized coaching feedback:

Player: {player_name if player_name else 'Unknown Player'}

SCORES (out of 100):
- Balance: {scores.get('balance', 0)}/100
- Footwork: {scores.get('footwork', 0)}/100
- Timing: {scores.get('timing', 0)}/100
- Bat Swing: {scores.get('bat_swing', 0)}/100
- Overall: {scores.get('overall', 0)}/100

WEAKNESSES: {weakness_text}
TOP BOWLING STRATEGY: {strategy_text}

Provide in under 200 words:
1. Brief overall assessment (reference actual scores)
2. Biggest strength and why
3. Most critical weakness and specific drill to fix it
4. One motivational tip

Be direct, specific, use cricket terminology."""

        response = _client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text

    except Exception as e:
        return f"AI analysis unavailable: {str(e)}"


def get_comparison_insight(player1_name, player1_scores, player2_name, player2_scores):
    try:
        prompt = f"""You are an expert cricket analyst comparing two batsmen.

PLAYER 1 — {player1_name}:
Balance: {player1_scores.get('balance',0)}, Footwork: {player1_scores.get('footwork',0)}, Timing: {player1_scores.get('timing',0)}, Bat Swing: {player1_scores.get('bat_swing',0)}, Overall: {player1_scores.get('overall',0)}

PLAYER 2 — {player2_name}:
Balance: {player2_scores.get('balance',0)}, Footwork: {player2_scores.get('footwork',0)}, Timing: {player2_scores.get('timing',0)}, Bat Swing: {player2_scores.get('bat_swing',0)}, Overall: {player2_scores.get('overall',0)}

In 150 words:
1. Who is technically stronger and why
2. What each player can learn from the other
3. One specific improvement tip for each

Reference actual scores, use cricket terminology."""

        response = _client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text

    except Exception as e:
        return f"AI comparison unavailable: {str(e)}"