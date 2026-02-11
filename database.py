"""
Database Module
Handles storage of predictions and recognition logs using SQLite
"""

import sqlite3
from datetime import datetime
import json
import os
import numpy as np

class RecognitionDatabase:
    def __init__(self, db_path='database/recognition_logs.db'):
        """
        Initialize database connection
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = None
        self.cursor = None
        self.connect()
        self.create_tables()
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            print(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
    
    def create_tables(self):
        """Create necessary tables if they don't exist"""
        # Predictions table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                predicted_digit INTEGER NOT NULL,
                confidence_score REAL NOT NULL,
                all_probabilities TEXT,
                image_source TEXT,
                processing_time REAL,
                user_feedback TEXT,
                true_label INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Model performance logs
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                accuracy REAL,
                precision_score REAL,
                recall_score REAL,
                f1_score REAL,
                total_predictions INTEGER,
                correct_predictions INTEGER,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # User sessions
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                total_predictions INTEGER DEFAULT 0,
                interface_type TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
        print("Database tables created/verified successfully")
    
    def log_prediction(self, predicted_digit, confidence_scores, 
                      image_source='unknown', processing_time=None,
                      true_label=None):
        """
        Log a single prediction
        
        Args:
            predicted_digit (int): Predicted digit (0-9)
            confidence_scores (array): Confidence scores for all classes
            image_source (str): Source of image (upload, webcam, drawn, etc.)
            processing_time (float): Processing time in seconds
            true_label (int): True label if known (for feedback)
            
        Returns:
            int: ID of inserted record
        """
        timestamp = datetime.now().isoformat()
        confidence_score = float(confidence_scores[predicted_digit])
        all_probs = json.dumps(confidence_scores.tolist())
        
        self.cursor.execute('''
            INSERT INTO predictions 
            (timestamp, predicted_digit, confidence_score, all_probabilities,
             image_source, processing_time, true_label)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, int(predicted_digit), confidence_score, all_probs,
              image_source, processing_time, true_label))
        
        self.conn.commit()
        return self.cursor.lastrowid
    
    def update_prediction_feedback(self, prediction_id, user_feedback, true_label=None):
        """
        Update prediction with user feedback
        
        Args:
            prediction_id (int): ID of prediction record
            user_feedback (str): User feedback text
            true_label (int): Correct label provided by user
        """
        self.cursor.execute('''
            UPDATE predictions
            SET user_feedback = ?, true_label = ?
            WHERE id = ?
        ''', (user_feedback, true_label, prediction_id))
        
        self.conn.commit()
    
    def log_model_performance(self, accuracy, precision, recall, f1_score,
                            total_predictions, correct_predictions, notes=''):
        """
        Log model performance metrics
        
        Args:
            accuracy (float): Model accuracy
            precision (float): Precision score
            recall (float): Recall score
            f1_score (float): F1 score
            total_predictions (int): Total number of predictions
            correct_predictions (int): Number of correct predictions
            notes (str): Additional notes
        """
        timestamp = datetime.now().isoformat()
        
        self.cursor.execute('''
            INSERT INTO model_performance
            (timestamp, accuracy, precision_score, recall_score, f1_score,
             total_predictions, correct_predictions, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, accuracy, precision, recall, f1_score,
              total_predictions, correct_predictions, notes))
        
        self.conn.commit()
    
    def create_session(self, session_id, interface_type='desktop'):
        """
        Create a new user session
        
        Args:
            session_id (str): Unique session identifier
            interface_type (str): Type of interface (desktop, web, mobile)
            
        Returns:
            int: Session record ID
        """
        start_time = datetime.now().isoformat()
        
        try:
            self.cursor.execute('''
                INSERT INTO user_sessions (session_id, start_time, interface_type)
                VALUES (?, ?, ?)
            ''', (session_id, start_time, interface_type))
            
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.IntegrityError:
            # Session already exists
            return None
    
    def end_session(self, session_id, total_predictions):
        """
        End a user session
        
        Args:
            session_id (str): Session identifier
            total_predictions (int): Total predictions made in session
        """
        end_time = datetime.now().isoformat()
        
        self.cursor.execute('''
            UPDATE user_sessions
            SET end_time = ?, total_predictions = ?
            WHERE session_id = ?
        ''', (end_time, total_predictions, session_id))
        
        self.conn.commit()
    
    def get_recent_predictions(self, limit=10):
        """
        Get recent predictions
        
        Args:
            limit (int): Number of recent records to fetch
            
        Returns:
            list: List of prediction records
        """
        self.cursor.execute('''
            SELECT * FROM predictions
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        
        columns = [description[0] for description in self.cursor.description]
        results = self.cursor.fetchall()
        
        return [dict(zip(columns, row)) for row in results]
    
    def get_statistics(self, days=7):
        """
        Get prediction statistics for the last N days
        
        Args:
            days (int): Number of days to look back
            
        Returns:
            dict: Statistics dictionary
        """
        # Total predictions
        self.cursor.execute('''
            SELECT COUNT(*) FROM predictions
            WHERE datetime(timestamp) >= datetime('now', ?)
        ''', (f'-{days} days',))
        total = self.cursor.fetchone()[0]
        
        # Predictions by digit
        self.cursor.execute('''
            SELECT predicted_digit, COUNT(*) as count
            FROM predictions
            WHERE datetime(timestamp) >= datetime('now', ?)
            GROUP BY predicted_digit
            ORDER BY predicted_digit
        ''', (f'-{days} days',))
        by_digit = dict(self.cursor.fetchall())
        
        # Accuracy (where true label is known)
        self.cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN predicted_digit = true_label THEN 1 ELSE 0 END) as correct
            FROM predictions
            WHERE true_label IS NOT NULL
            AND datetime(timestamp) >= datetime('now', ?)
        ''', (f'-{days} days',))
        accuracy_data = self.cursor.fetchone()
        accuracy = (accuracy_data[1] / accuracy_data[0] * 100) if accuracy_data[0] > 0 else 0
        
        # Average confidence
        self.cursor.execute('''
            SELECT AVG(confidence_score)
            FROM predictions
            WHERE datetime(timestamp) >= datetime('now', ?)
        ''', (f'-{days} days',))
        avg_confidence = self.cursor.fetchone()[0] or 0
        
        # By source
        self.cursor.execute('''
            SELECT image_source, COUNT(*) as count
            FROM predictions
            WHERE datetime(timestamp) >= datetime('now', ?)
            GROUP BY image_source
        ''', (f'-{days} days',))
        by_source = dict(self.cursor.fetchall())
        
        return {
            'total_predictions': total,
            'predictions_by_digit': by_digit,
            'accuracy': accuracy,
            'average_confidence': avg_confidence,
            'predictions_by_source': by_source,
            'days': days
        }
    
    def export_to_json(self, output_path='database/export.json'):
        """
        Export all data to JSON file
        
        Args:
            output_path (str): Path to output JSON file
        """
        data = {
            'predictions': self.get_recent_predictions(limit=1000),
            'statistics': self.get_statistics(days=30)
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"Data exported to: {output_path}")
    
    def clear_old_records(self, days=30):
        """
        Clear records older than N days
        
        Args:
            days (int): Keep records newer than this many days
        """
        self.cursor.execute('''
            DELETE FROM predictions
            WHERE datetime(timestamp) < datetime('now', ?)
        ''', (f'-{days} days',))
        
        deleted = self.cursor.rowcount
        self.conn.commit()
        
        print(f"Deleted {deleted} old records")
        return deleted
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("Database connection closed")


if __name__ == "__main__":
    print("=== Testing Database Module ===\n")
    
    # Create database instance
    db = RecognitionDatabase(db_path='database/test_recognition.db')
    
    # Test logging prediction
    print("1. Testing prediction logging...")
    confidence_scores = np.array([0.05, 0.03, 0.02, 0.85, 0.01, 0.01, 0.01, 0.01, 0.01, 0.00])
    prediction_id = db.log_prediction(
        predicted_digit=3,
        confidence_scores=confidence_scores,
        image_source='test',
        processing_time=0.15
    )
    print(f"   Logged prediction with ID: {prediction_id}")
    
    # Test multiple predictions
    print("\n2. Logging multiple test predictions...")
    for i in range(5):
        digit = np.random.randint(0, 10)
        scores = np.random.random(10)
        scores = scores / scores.sum()  # Normalize
        db.log_prediction(
            predicted_digit=digit,
            confidence_scores=scores,
            image_source='test_batch'
        )
    print("   Logged 5 test predictions")
    
    # Test model performance logging
    print("\n3. Testing model performance logging...")
    db.log_model_performance(
        accuracy=0.985,
        precision=0.987,
        recall=0.983,
        f1_score=0.985,
        total_predictions=1000,
        correct_predictions=985,
        notes='Test run'
    )
    print("   Logged model performance")
    
    # Test session management
    print("\n4. Testing session management...")
    session_id = 'test_session_001'
    db.create_session(session_id, interface_type='test')
    db.end_session(session_id, total_predictions=6)
    print(f"   Created and ended session: {session_id}")
    
    # Get recent predictions
    print("\n5. Fetching recent predictions...")
    recent = db.get_recent_predictions(limit=3)
    print(f"   Retrieved {len(recent)} recent predictions")
    for pred in recent:
        print(f"     Digit: {pred['predicted_digit']}, "
              f"Confidence: {pred['confidence_score']:.3f}")
    
    # Get statistics
    print("\n6. Getting statistics...")
    stats = db.get_statistics(days=7)
    print(f"   Total predictions (7 days): {stats['total_predictions']}")
    print(f"   Average confidence: {stats['average_confidence']:.3f}")
    
    # Export data
    print("\n7. Exporting data to JSON...")
    db.export_to_json('database/test_export.json')
    
    # Close connection
    print("\n8. Closing database connection...")
    db.close()
    
    print("\n✓ Database module tested successfully!")
