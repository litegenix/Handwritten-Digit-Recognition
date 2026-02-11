"""
Desktop GUI Application using Tkinter
Provides interface for drawing digits and uploading images for recognition
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import cv2
from datetime import datetime
import os

class DigitRecognitionGUI:
    def __init__(self, model, preprocessor, database):
        """
        Initialize GUI
        
        Args:
            model: Trained CNN model instance
            preprocessor: Data preprocessor instance
            database: Database instance for logging
        """
        self.model = model
        self.preprocessor = preprocessor
        self.database = database
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Handwritten Digit Recognition System")
        self.root.geometry("1000x700")
        self.root.resizable(False, False)
        
        # Drawing canvas variables
        self.canvas_size = 280
        self.brush_size = 15
        self.drawing = False
        self.last_x = 0
        self.last_y = 0
        
        # Image for drawing
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 'black')
        self.draw = ImageDraw.Draw(self.image)
        
        # Prediction counter
        self.prediction_count = 0
        
        # Session ID
        self.session_id = f"desktop_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.database.create_session(self.session_id, interface_type='desktop')
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup GUI components"""
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="✍ Handwritten Digit Recognition",
            font=('Arial', 24, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=20)
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg='#ecf0f1')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Drawing area
        left_panel = tk.Frame(main_frame, bg='#ecf0f1')
        left_panel.pack(side=tk.LEFT, padx=10)
        
        # Drawing canvas
        canvas_label = tk.Label(
            left_panel,
            text="Draw a digit (0-9):",
            font=('Arial', 14, 'bold'),
            bg='#ecf0f1'
        )
        canvas_label.pack(pady=10)
        
        self.canvas = tk.Canvas(
            left_panel,
            width=self.canvas_size,
            height=self.canvas_size,
            bg='black',
            cursor='cross'
        )
        self.canvas.pack()
        
        # Bind mouse events for drawing
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw_digit)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)
        
        # Control buttons (drawing)
        draw_button_frame = tk.Frame(left_panel, bg='#ecf0f1')
        draw_button_frame.pack(pady=15)
        
        self.predict_btn = tk.Button(
            draw_button_frame,
            text="🔍 Recognize",
            font=('Arial', 12, 'bold'),
            bg='#27ae60',
            fg='white',
            width=12,
            height=2,
            command=self.predict_drawn_digit,
            cursor='hand2'
        )
        self.predict_btn.grid(row=0, column=0, padx=5)
        
        self.clear_btn = tk.Button(
            draw_button_frame,
            text="🗑 Clear",
            font=('Arial', 12, 'bold'),
            bg='#e74c3c',
            fg='white',
            width=12,
            height=2,
            command=self.clear_canvas,
            cursor='hand2'
        )
        self.clear_btn.grid(row=0, column=1, padx=5)
        
        # Upload button
        self.upload_btn = tk.Button(
            left_panel,
            text="📁 Upload Image",
            font=('Arial', 12, 'bold'),
            bg='#3498db',
            fg='white',
            width=26,
            height=2,
            command=self.upload_image,
            cursor='hand2'
        )
        self.upload_btn.pack(pady=10)
        
        # Middle panel - Preview and controls
        middle_panel = tk.Frame(main_frame, bg='#ecf0f1')
        middle_panel.pack(side=tk.LEFT, padx=20)
        
        # Preview label
        preview_label = tk.Label(
            middle_panel,
            text="Preprocessed (28x28):",
            font=('Arial', 12, 'bold'),
            bg='#ecf0f1'
        )
        preview_label.pack(pady=10)
        
        # Preview canvas
        self.preview_canvas = tk.Canvas(
            middle_panel,
            width=140,
            height=140,
            bg='white',
            highlightthickness=2,
            highlightbackground='#34495e'
        )
        self.preview_canvas.pack()
        
        # Brush size control
        brush_frame = tk.Frame(middle_panel, bg='#ecf0f1')
        brush_frame.pack(pady=20)
        
        brush_label = tk.Label(
            brush_frame,
            text="Brush Size:",
            font=('Arial', 11),
            bg='#ecf0f1'
        )
        brush_label.pack()
        
        self.brush_slider = tk.Scale(
            brush_frame,
            from_=5,
            to=30,
            orient=tk.HORIZONTAL,
            length=150,
            command=self.update_brush_size
        )
        self.brush_slider.set(self.brush_size)
        self.brush_slider.pack()
        
        # Statistics
        stats_frame = tk.LabelFrame(
            middle_panel,
            text="Session Statistics",
            font=('Arial', 11, 'bold'),
            bg='#ecf0f1',
            padx=10,
            pady=10
        )
        stats_frame.pack(pady=20, fill=tk.X)
        
        self.stats_label = tk.Label(
            stats_frame,
            text="Predictions: 0",
            font=('Arial', 10),
            bg='#ecf0f1',
            justify=tk.LEFT
        )
        self.stats_label.pack()
        
        # Right panel - Results
        right_panel = tk.Frame(main_frame, bg='#ecf0f1', width=300)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Result display
        result_label = tk.Label(
            right_panel,
            text="Recognition Result:",
            font=('Arial', 14, 'bold'),
            bg='#ecf0f1'
        )
        result_label.pack(pady=10)
        
        # Predicted digit display
        self.result_frame = tk.Frame(right_panel, bg='white', 
                                     highlightthickness=2,
                                     highlightbackground='#34495e')
        self.result_frame.pack(pady=10)
        
        self.digit_label = tk.Label(
            self.result_frame,
            text="?",
            font=('Arial', 80, 'bold'),
            bg='white',
            fg='#2c3e50',
            width=3,
            height=2
        )
        self.digit_label.pack(padx=20, pady=20)
        
        # Confidence display
        self.confidence_label = tk.Label(
            right_panel,
            text="Confidence: ---%",
            font=('Arial', 14, 'bold'),
            bg='#ecf0f1',
            fg='#27ae60'
        )
        self.confidence_label.pack(pady=10)
        
        # Probability bars
        prob_label = tk.Label(
            right_panel,
            text="All Probabilities:",
            font=('Arial', 12, 'bold'),
            bg='#ecf0f1'
        )
        prob_label.pack(pady=5)
        
        self.prob_frame = tk.Frame(right_panel, bg='#ecf0f1')
        self.prob_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        
        self.prob_bars = []
        for i in range(10):
            bar_frame = tk.Frame(self.prob_frame, bg='#ecf0f1')
            bar_frame.pack(fill=tk.X, pady=2)
            
            label = tk.Label(bar_frame, text=f"{i}:", font=('Arial', 10),
                           bg='#ecf0f1', width=2)
            label.pack(side=tk.LEFT)
            
            canvas = tk.Canvas(bar_frame, height=20, bg='#bdc3c7',
                             highlightthickness=0)
            canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            percent_label = tk.Label(bar_frame, text="0%", font=('Arial', 9),
                                   bg='#ecf0f1', width=6)
            percent_label.pack(side=tk.LEFT)
            
            self.prob_bars.append((canvas, percent_label))
    
    def start_draw(self, event):
        """Start drawing"""
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
    
    def draw_digit(self, event):
        """Draw on canvas"""
        if self.drawing:
            x, y = event.x, event.y
            
            # Draw on canvas
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                fill='white',
                width=self.brush_size,
                capstyle=tk.ROUND,
                smooth=True
            )
            
            # Draw on PIL image
            self.draw.line(
                [(self.last_x, self.last_y), (x, y)],
                fill='white',
                width=self.brush_size
            )
            
            self.last_x = x
            self.last_y = y
    
    def stop_draw(self, event):
        """Stop drawing"""
        self.drawing = False
    
    def update_brush_size(self, value):
        """Update brush size"""
        self.brush_size = int(value)
    
    def clear_canvas(self):
        """Clear drawing canvas"""
        self.canvas.delete('all')
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 'black')
        self.draw = ImageDraw.Draw(self.image)
        self.preview_canvas.delete('all')
        self.reset_results()
    
    def reset_results(self):
        """Reset result displays"""
        self.digit_label.config(text="?", fg='#2c3e50')
        self.confidence_label.config(text="Confidence: ---%")
        for canvas, label in self.prob_bars:
            canvas.delete('all')
            label.config(text="0%")
    
    def predict_drawn_digit(self):
        """Predict digit from drawn canvas"""
        try:
            # Get image array
            img_array = np.array(self.image)
            
            # Check if canvas is empty
            if img_array.max() == 0:
                messagebox.showwarning("Warning", "Please draw a digit first!")
                return
            
            # Preprocess
            start_time = datetime.now()
            preprocessed = self.preprocessor.preprocess_drawn_digit(img_array)
            
            # Show preview
            self.show_preview(preprocessed)
            
            # Predict
            predicted_digit, confidence = self.model.predict(preprocessed)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Log to database
            self.database.log_prediction(
                predicted_digit=predicted_digit,
                confidence_scores=confidence,
                image_source='drawn',
                processing_time=processing_time
            )
            
            # Update displays
            self.update_results(predicted_digit, confidence)
            
            # Update statistics
            self.prediction_count += 1
            self.stats_label.config(text=f"Predictions: {self.prediction_count}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def upload_image(self):
        """Upload and predict from image file"""
        filepath = filedialog.askopenfilename(
            title="Select Digit Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        
        if not filepath:
            return
        
        try:
            # Load and preprocess
            start_time = datetime.now()
            preprocessed = self.preprocessor.preprocess_image(filepath)
            
            # Show preview
            self.show_preview(preprocessed)
            
            # Predict
            predicted_digit, confidence = self.model.predict(preprocessed)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Log to database
            self.database.log_prediction(
                predicted_digit=predicted_digit,
                confidence_scores=confidence,
                image_source='uploaded',
                processing_time=processing_time
            )
            
            # Update displays
            self.update_results(predicted_digit, confidence)
            
            # Update statistics
            self.prediction_count += 1
            self.stats_label.config(text=f"Predictions: {self.prediction_count}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
    
    def show_preview(self, preprocessed_image):
        """Show preprocessed image preview"""
        # Convert to displayable format
        img = (preprocessed_image.squeeze() * 255).astype(np.uint8)
        img = cv2.resize(img, (140, 140), interpolation=cv2.INTER_NEAREST)
        
        # Convert to PhotoImage
        pil_img = Image.fromarray(img)
        photo = ImageTk.PhotoImage(pil_img)
        
        # Display
        self.preview_canvas.delete('all')
        self.preview_canvas.create_image(70, 70, image=photo)
        self.preview_canvas.image = photo  # Keep reference
    
    def update_results(self, digit, confidence):
        """Update result displays"""
        # Update digit display
        self.digit_label.config(text=str(digit), fg='#27ae60')
        
        # Update confidence
        conf_percent = confidence[digit] * 100
        self.confidence_label.config(
            text=f"Confidence: {conf_percent:.1f}%",
            fg='#27ae60' if conf_percent > 80 else '#f39c12'
        )
        
        # Update probability bars
        for i, (canvas, label) in enumerate(self.prob_bars):
            prob = confidence[i] * 100
            
            # Clear canvas
            canvas.delete('all')
            
            # Draw bar
            bar_width = int((prob / 100) * (canvas.winfo_width() - 4))
            color = '#27ae60' if i == digit else '#3498db'
            canvas.create_rectangle(
                2, 2, bar_width, 18,
                fill=color,
                outline=''
            )
            
            # Update label
            label.config(text=f"{prob:.1f}%")
    
    def run(self):
        """Start GUI application"""
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start main loop
        self.root.mainloop()
    
    def on_closing(self):
        """Handle window close event"""
        # End session in database
        self.database.end_session(self.session_id, self.prediction_count)
        
        # Close database connection
        self.database.close()
        
        # Destroy window
        self.root.destroy()


if __name__ == "__main__":
    print("Desktop GUI requires trained model.")
    print("Please run train_model.py first to train the model.")
    print("\nThen run: python desktop_app.py")
