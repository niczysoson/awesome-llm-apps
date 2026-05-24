"""AI Chess Agent using LLM to play chess against a human opponent.

This module implements a chess-playing AI agent that uses an LLM (Claude/GPT)
to analyze board positions and make strategic moves via the python-chess library.
"""

import os
import chess
import chess.svg
import streamlit as st
from anthropic import Anthropic

# Initialize Anthropic client
client = Anthropic()

SYSTEM_PROMPT = """You are an expert chess player with deep knowledge of chess strategy,
tactics, and endgame theory. You are playing as {color} pieces.

When given a chess position in FEN notation, analyze the position and suggest the best move.
You must respond with ONLY a valid UCI move (e.g., 'e2e4', 'g1f3', 'e1g1' for castling).
Do not include any explanation or additional text — just the UCI move string.

Always ensure the move is legal for the current position."""


def get_ai_move(board: chess.Board, ai_color: str) -> str:
    """Get the AI's next move using the LLM.

    Args:
        board: Current chess board state.
        ai_color: Color the AI is playing ('white' or 'black').

    Returns:
        UCI move string chosen by the AI.
    """
    fen = board.fen()
    legal_moves = [move.uci() for move in board.legal_moves]

    user_message = (
        f"Current board position (FEN): {fen}\n"
        f"Legal moves available: {', '.join(legal_moves)}\n"
        "Choose the best move from the legal moves list and respond with only the UCI move string."
    )

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=16,
        system=SYSTEM_PROMPT.format(color=ai_color),
        messages=[{"role": "user", "content": user_message}],
    )

    move_str = response.content[0].text.strip()
    return move_str


def render_board(board: chess.Board) -> str:
    """Render the chess board as an SVG string.

    Args:
        board: Current chess board state.

    Returns:
        SVG string representation of the board.
    """
    last_move = board.peek() if board.move_stack else None
    svg = chess.svg.board(
        board,
        lastmove=last_move,
        size=400,
        colors={"square light": "#f0d9b5", "square dark": "#b58863"},
    )
    return svg


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "board" not in st.session_state:
        st.session_state.board = chess.Board()
    if "game_over" not in st.session_state:
        st.session_state.game_over = False
    if "status_message" not in st.session_state:
        st.session_state.status_message = "Your turn! Enter a move in UCI format (e.g., e2e4)."
    if "move_history" not in st.session_state:
        st.session_state.move_history = []


def main():
    """Main Streamlit application entry point."""
    st.set_page_config(page_title="AI Chess Agent", page_icon="♟️", layout="centered")
    st.title("♟️ AI Chess Agent")
    st.markdown("Play chess against Claude AI. You play as **White**, AI plays as **Black**.")

    initialize_session_state()

    board: chess.Board = st.session_state.board

    # Render the board
    board_svg = render_board(board)
    st.image(board_svg.encode(), use_container_width=False, width=420)

    # Status message
    st.info(st.session_state.status_message)

    # Move history
    if st.session_state.move_history:
        with st.expander("Move History", expanded=False):
            pairs = [
                f"{i + 1}. {st.session_state.move_history[i * 2]} "
                + (st.session_state.move_history[i * 2 + 1] if i * 2 + 1 < len(st.session_state.move_history) else "")
                for i in range((len(st.session_state.move_history) + 1) // 2)
            ]
            st.text("\n".join(pairs))

    # Player input
    if not st.session_state.game_over and board.turn == chess.WHITE:
        with st.form(key="move_form", clear_on_submit=True):
            user_move = st.text_input("Your move (UCI format, e.g., e2e4):", max_chars=5)
            submitted = st.form_submit_button("Make Move")

        if submitted and user_move:
            try:
                move = chess.Move.from_uci(user_move.strip().lower())
                if move in board.legal_moves:
                    board.push(move)
                    st.session_state.move_history.append(user_move.strip().lower())

                    if board.is_game_over():
                        st.session_state.game_over = True
                        st.session_state.status_message = f"Game over! Result: {board.result()}"
                    else:
                        # AI makes its move
                        st.session_state.status_message = "AI is thinking..."
                        ai_uci = get_ai_move(board, "black")
                        ai_move = chess.Move.from_uci(ai_uci)
                        if ai_move in board.legal_moves:
                            board.push(ai_move)
                            st.session_state.move_history.append(ai_uci)
                            if board.is_game_over():
                                st.session_state.game_over = True
                                st.session_state.status_message = f"Game over! Result: {board.result()}"
                            else:
                                st.session_state.status_message = f"AI played {ai_uci}. Your turn!"
                        else:
                            st.session_state.status_message = f"AI returned invalid move '{ai_uci}'. Please try again."
                    st.rerun()
                else:
                    st.error(f"Illegal move: {user_move}. Please enter a valid UCI move.")
            except ValueError:
                st.error("Invalid move format. Use UCI notation like 'e2e4' or 'g1f3'.")

    # Reset button
    if st.button("🔄 New Game"):
        st.session_state.board = chess.Board()
        st.session_state.game_over = False
        st.session_state.status_message = "New game started! Your turn."
        st.session_state.move_history = []
        st.rerun()


if __name__ == "__main__":
    main()
