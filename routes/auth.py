from flask import Blueprint, render_template, request, redirect, url_for, session
import mysql.connector
from werkzeug.security import check_password_hash, generate_password_hash

from db.connection import get_db_connection, is_duplicate_entry_error

auth_bp = Blueprint("auth", __name__)


def require_login():
    return "user_id" in session


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        email = request.form.get("email", "")
        password = request.form.get("password", "")

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, password_hash, first_name FROM users WHERE email = %s",
            (email,),
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if row and check_password_hash(row[1], password):
            session["user_id"] = row[0]
            session["email"] = email
            session["display_name"] = row[2] or email
            return redirect(url_for("main.main"))
        error = "Invalid email or password."

    return render_template("login.html", error=error)


@auth_bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("auth.login"))


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if require_login():
        return redirect(url_for("main.main"))
    error = None
    if request.method == "POST":
        first_name = request.form.get("first_name", "")
        last_name = request.form.get("last_name", "")
        email = request.form.get("email", "")
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not email or not password:
            error = "Email and password are required."
        elif password != confirm_password:
            error = "Passwords do not match."
        else:
            password_hash = generate_password_hash(password)
            conn = get_db_connection()
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    INSERT INTO users (email, password_hash, first_name, last_name)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (email, password_hash, first_name, last_name),
                )
                conn.commit()
                session["user_id"] = cursor.lastrowid
                session["email"] = email
                session["display_name"] = first_name or email
                return redirect(url_for("main.main"))
            except mysql.connector.Error as exc:
                conn.rollback()
                if is_duplicate_entry_error(exc):
                    error = "Email already exists."
                else:
                    error = "Registration failed."
            finally:
                cursor.close()
                conn.close()

    return render_template("register.html", error=error)
