from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from bson import ObjectId
from datetime import datetime
import hashlib

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "image_colorization"

def get_db():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    return client[DB_NAME]

def init_db():
    try:
        db = get_db()
        db.users.create_index("username", unique=True)
        # Create default admin if not exists
        if not db.users.find_one({"username": "admin"}):
            db.users.insert_one({
                "username": "admin",
                "password": hashlib.sha256("admin123".encode()).hexdigest(),
                "email": "admin@colorize.ai",
                "role": "admin",
                "is_active": True,
                "created_at": datetime.utcnow()
            })
            print("Default admin created — username: admin, password: admin123")
        print("MongoDB connected and initialized.")
    except ServerSelectionTimeoutError:
        print("WARNING: MongoDB not reachable. Start MongoDB service and restart the app.")

# ── User functions ────────────────────────────────────────────────────────────

def create_user(username, password, email=None):
    try:
        db = get_db()
        db.users.insert_one({
            "username": username,
            "password": password,
            "email": email,
            "role": "user",
            "is_active": True,
            "created_at": datetime.utcnow()
        })
        return True
    except Exception:
        return False

def verify_user(username, password):
    try:
        db = get_db()
        user = db.users.find_one({"username": username, "password": password, "is_active": True})
        if user:
            return (str(user["_id"]), user["username"], user.get("role", "user"))
    except Exception:
        pass
    return None

def save_colorization(user_id, original_filename, original_path, colorized_path):
    try:
        db = get_db()
        db.colorizations.insert_one({
            "user_id": user_id,
            "original_filename": original_filename,
            "original_path": original_path,
            "colorized_path": colorized_path,
            "created_at": datetime.utcnow()
        })
    except Exception:
        pass

def get_user_colorizations(user_id):
    try:
        db = get_db()
        results = db.colorizations.find({"user_id": user_id}).sort("created_at", -1)
        return [
            (str(r["_id"]), r["original_filename"], r["original_path"], r["colorized_path"], r["created_at"])
            for r in results
        ]
    except Exception:
        return []

# ── Admin functions ───────────────────────────────────────────────────────────

def get_all_users():
    try:
        db = get_db()
        users = db.users.find().sort("created_at", -1)
        result = []
        for u in users:
            count = db.colorizations.count_documents({"user_id": str(u["_id"])})
            result.append({
                "id": str(u["_id"]),
                "username": u["username"],
                "email": u.get("email", "—"),
                "role": u.get("role", "user"),
                "is_active": u.get("is_active", True),
                "created_at": u.get("created_at"),
                "colorization_count": count
            })
        return result
    except Exception:
        return []

def get_admin_stats():
    try:
        db = get_db()
        total_users = db.users.count_documents({"role": {"$ne": "admin"}})
        total_colorizations = db.colorizations.count_documents({})
        active_users = db.users.count_documents({"is_active": True, "role": {"$ne": "admin"}})
        # colorizations today
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today_colorizations = db.colorizations.count_documents({"created_at": {"$gte": today}})
        return {
            "total_users": total_users,
            "active_users": active_users,
            "total_colorizations": total_colorizations,
            "today_colorizations": today_colorizations
        }
    except Exception:
        return {"total_users": 0, "active_users": 0, "total_colorizations": 0, "today_colorizations": 0}

def get_recent_activity(limit=10):
    try:
        db = get_db()
        results = db.colorizations.find().sort("created_at", -1).limit(limit)
        activity = []
        for r in results:
            user = db.users.find_one({"_id": ObjectId(r["user_id"])}) if ObjectId.is_valid(r["user_id"]) else None
            activity.append({
                "id": str(r["_id"]),
                "username": user["username"] if user else "Unknown",
                "filename": r["original_filename"],
                "created_at": r["created_at"]
            })
        return activity
    except Exception:
        return []

def toggle_user_status(user_id):
    try:
        db = get_db()
        user = db.users.find_one({"_id": ObjectId(user_id)})
        if user:
            new_status = not user.get("is_active", True)
            db.users.update_one({"_id": ObjectId(user_id)}, {"$set": {"is_active": new_status}})
            return new_status
    except Exception:
        pass
    return None

def delete_user(user_id):
    try:
        db = get_db()
        db.users.delete_one({"_id": ObjectId(user_id)})
        db.colorizations.delete_many({"user_id": user_id})
        return True
    except Exception:
        return False

def update_user_role(user_id, role):
    try:
        db = get_db()
        db.users.update_one({"_id": ObjectId(user_id)}, {"$set": {"role": role}})
        return True
    except Exception:
        return False

if __name__ == "__main__":
    init_db()
