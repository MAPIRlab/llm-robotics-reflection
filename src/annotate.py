import json
import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import yaml

import constants
from utils import file_utils
from voxelad import preprocess


class GUI(tk.Tk):
    def __init__(self):
        super().__init__()

        # Flag to check if semantic map JSON has been loaded
        self.semantic_map_loaded = False
        # Flag to check if queries YAML file has been loaded
        self.queries_loaded = False
        # Variable to store the responses JSON file path
        self.responses_file_path = None

        self.title("Semantic Map and Human Data Interface")
        self.geometry("1200x800")

        # Create a PanedWindow for resizable left and right sections
        self.paned_window = tk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # Left Frame (Semantic Map)
        self.left_frame = tk.Frame(self.paned_window, bg='lightblue')
        self.paned_window.add(self.left_frame)

        self.left_label = tk.Label(
            self.left_frame, text="Semantic map", font=("Arial", 12, "bold"))
        self.left_label.pack(pady=10)

        # "Load semantic map JSON file"
        self.load_semantic_map_button = tk.Button(
            self.left_frame, text="Load semantic map JSON file", command=self.load_semantic_map)
        self.load_semantic_map_button.pack(pady=5)

        # Loaded semantic map file label
        self.semantic_map_label = tk.Label(
            self.left_frame, text="No semantic map file loaded", font=("Arial", 10, "italic"))
        self.semantic_map_label.pack(pady=5)

        self.semantic_map_tree = ttk.Treeview(self.left_frame, columns=(
            "Object id", "Most Probable Object", "BoundingBoxCenter", "BoundingBoxSize", "Results"), show='headings', height=10)
        self.semantic_map_tree.heading("Object id", text="Object id")
        self.semantic_map_tree.heading(
            "Most Probable Object", text="Most Probable Object")
        self.semantic_map_tree.heading(
            "BoundingBoxCenter", text="Bounding Box Center")
        self.semantic_map_tree.heading(
            "BoundingBoxSize", text="Bounding Box Size")
        self.semantic_map_tree.heading("Results", text="Results")
        self.semantic_map_tree.pack(
            padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Right Frame (Human Data)
        self.right_frame = tk.Frame(self.paned_window, bg='lightyellow')
        self.paned_window.add(self.right_frame)

        self.right_label = tk.Label(
            self.right_frame, text="Human data", font=("Arial", 12, "bold"))
        self.right_label.pack(pady=10)

        self.load_query_button = tk.Button(
            self.right_frame, text="Load query YAML file", command=self.load_query_file)
        self.load_query_button.pack(pady=5)

        # "Load responses JSON file" button
        self.load_responses_button = tk.Button(
            self.right_frame, text="Load responses JSON file", command=self.load_responses_file)
        self.load_responses_button.pack(pady=5)

        # Label to show the loaded responses file name
        self.responses_label = tk.Label(
            self.right_frame, text="No responses file loaded", font=("Arial", 10, "italic"))
        self.responses_label.pack(pady=5)

        self.human_data_tree = ttk.Treeview(self.right_frame, columns=(
            "Query ID", "Query", "Response"), show='headings', height=10)
        self.human_data_tree.heading("Query ID", text="Query ID")
        self.human_data_tree.heading("Query", text="Query")
        self.human_data_tree.heading("Response", text="Response")
        self.human_data_tree.column("Response", width=200)
        self.human_data_tree.bind("<Double-1>", self.on_double_click)
        self.human_data_tree.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.human_data_tree.bind(
            "<<TreeviewSelect>>", self.highlight_semantic_map_objects)

        # # Save Button at the bottom, occupying the entire width
        # self.save_button = tk.Button(
        #     self, text="Save", command=self.save_data, bg="blue", fg="white")
        # self.save_button.pack(side=tk.BOTTOM, fill=tk.X)

        self.clear_button = tk.Button(
            self.right_frame, text="Clear", command=self.clear_queries_and_responses, bg="red", fg="white")
        self.clear_button.pack(pady=5)

    def load_semantic_map(self):
        file_path = filedialog.askopenfilename(
            title="Select a semantic map JSON file",
            filetypes=[("JSON files", "*.json")],
            initialdir=constants.SEMANTIC_MAPS_FOLDER_PATH
        )
        if file_path:

            # Clear loaded queries and responses
            self.clear_queries_and_responses()

            try:
                # Attempt to load the JSON file
                with open(file_path, 'r') as file:
                    semantic_map_obj = json.load(file)
                    semantic_map_obj = preprocess.preprocess_semantic_map(
                        semantic_map_obj, True)

                # Validate the structure of the JSON file
                if "instances" not in semantic_map_obj or not isinstance(semantic_map_obj["instances"], dict):
                    raise ValueError(
                        "The 'instances' key is missing or is not a dictionary.")

                # Clear the table before loading new data
                for row in self.semantic_map_tree.get_children():
                    self.semantic_map_tree.delete(row)

                # Sort the objects using natural sorting by extracting numbers
                def natural_sort_key(obj_id):
                    # Extract numeric parts for natural sorting
                    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', obj_id)]

                sorted_obj_ids = sorted(
                    semantic_map_obj["instances"].keys(), key=natural_sort_key)

                # Load each sorted object into the table
                for obj_id in sorted_obj_ids:
                    obj_data = semantic_map_obj["instances"][obj_id]

                    # Ensure the object has the required 'bbox' and 'results' structure
                    if "bbox" not in obj_data or "center" not in obj_data["bbox"] or "results" not in obj_data:
                        raise ValueError(f"Invalid structure for object '{
                                         obj_id}'. Missing 'bbox' or 'results'.")

                    bbox_size = tuple(obj_data["bbox"]["size"])
                    bbox_center = tuple(obj_data["bbox"]["center"])

                    # Sort the results by score in descending order and format them as a string
                    sorted_results = sorted(
                        obj_data["results"].items(), key=lambda x: x[1], reverse=True)
                    results = ", ".join(
                        [f"{label}: {score:.2f}" for label, score in sorted_results])

                    # Calculate the most probable object and its normalized score
                    total_score = sum(score for _, score in sorted_results)
                    if total_score > 0:
                        most_probable_object = sorted_results[0][0]
                        most_probable_score = sorted_results[0][1]
                        normalized_score = most_probable_score / total_score
                        most_probable_display = f"{
                            most_probable_object} ({normalized_score:.2f})"
                    else:
                        most_probable_display = "unknown (0.00)"

                    # Insert object data into the table with the new column order
                    self.semantic_map_tree.insert(
                        "", "end", values=(obj_id, most_probable_display, bbox_center, bbox_size, results))

                # Set the flag to True after successfully loading the semantic map
                self.semantic_map_loaded = True

                # Display the base name of the loaded semantic map file
                self.semantic_map_label.config(
                    text=f"Loaded file: {os.path.basename(file_path)}")

            except (json.JSONDecodeError, ValueError) as e:
                # If any JSON or structural error occurs, display the specific error message
                messagebox.showerror(
                    "Error", f"Semantic map is not in the desired format! Error: {str(e)}")

    def load_query_file(self):
        if not self.semantic_map_loaded:  # Check if the semantic map has been loaded
            messagebox.showerror("Error", "Load a semantic map file first!")
            return

        file_path = filedialog.askopenfilename(
            title="Select a query YAML file",
            filetypes=[("YAML files", "*.yaml")],
            initialdir=constants.DATA_FOLDER_PATH
        )
        if file_path:
            try:
                # Attempt to load the YAML file
                data = file_utils.load_yaml(file_path)

                # Validate the structure of the YAML file
                if "queries" not in data or not isinstance(data["queries"], dict):
                    raise ValueError(
                        "The 'queries' key is missing or is not a dictionary.")

                # Clear the table before loading new data
                for row in self.human_data_tree.get_children():
                    self.human_data_tree.delete(row)

                # Load each query into the right table
                for query_id, query_text in data["queries"].items():
                    # Leave Response empty for user input
                    self.human_data_tree.insert(
                        "", "end", values=(query_id, query_text, ""))

                # Set the flag to True after successfully loading the queries
                self.queries_loaded = True

            except (yaml.YAMLError, ValueError) as e:
                # If any YAML or structural error occurs, display the specific error message
                messagebox.showerror(
                    "Error", f"Queries file is not in the desired format! Error: {str(e)}")

    def on_double_click(self, event):
        # Detect the row and column where the double-click occurred
        item_id = self.human_data_tree.focus()
        column = self.human_data_tree.identify_column(event.x)

        if column == '#3':  # If it's the Response column
            # Get the bounding box of the selected cell
            bbox = self.human_data_tree.bbox(item_id, column)

            if bbox:  # Ensure bbox is not None
                # Get the current value of the Response cell
                item = self.human_data_tree.item(item_id)
                current_value = item['values'][2]

                # If an Entry widget already exists, destroy it
                if hasattr(self, 'current_entry_widget') and self.current_entry_widget is not None:
                    self.current_entry_widget.destroy()

                # Create an Entry widget to edit the cell
                self.current_entry_widget = tk.Entry(self.human_data_tree)
                self.current_entry_widget.insert(0, current_value)
                self.current_entry_widget.bind("<Return>", lambda e: self.save_response(
                    item_id, self.current_entry_widget.get()))
                self.current_entry_widget.place(
                    x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3])

                self.current_entry_widget.focus()

    def save_response(self, item_id, new_value):
        # Validate the response format
        try:
            # Parse the response as a JSON list or treat as empty list if blank
            if not new_value.strip():
                response_list = []
            else:
                response_list = [item.strip()
                                 for item in new_value.split(',') if item.strip()]

            # Ensure response_list is in JSON format
            json.dumps(response_list)

        except (json.JSONDecodeError, ValueError):
            messagebox.showerror("Error", "Wrong format! Introduce a list.")
            return

        # Validate each object ID in the response list
        existing_object_ids = {self.semantic_map_tree.item(
            row, "values")[0] for row in self.semantic_map_tree.get_children()}
        seen_ids = set()

        for obj_id in response_list:
            if obj_id in seen_ids:
                messagebox.showerror("Error", f"The object {
                                     obj_id} appears several times.")
                return
            if obj_id not in existing_object_ids:
                messagebox.showerror("Error", f"The object {
                                     obj_id} does not exist in the semantic map.")
                return
            seen_ids.add(obj_id)

        # If validation is successful, save the response in the table
        self.human_data_tree.item(item_id, values=(
            self.human_data_tree.item(item_id)["values"][0],  # Query ID
            self.human_data_tree.item(item_id)["values"][1],  # Query text
            ", ".join(response_list)))  # Save as a comma-separated string

        # Destroy the Entry widget after saving
        self.current_entry_widget.destroy()
        self.current_entry_widget = None

        # Update highlights based on the new Response value
        self.highlight_semantic_map_objects(None)

        # Set focus back to the right table
        self.human_data_tree.focus_set()

        # Check if a responses file path is set; if not, prompt for save location
        if not self.responses_file_path:
            file_path = filedialog.asksaveasfilename(
                title="Save Responses File",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")]
            )
            if file_path:
                self.responses_file_path = file_path  # Store the selected file path
                # Update the label to display the new file path
                self.responses_label.config(
                    text=f"Loaded file: {os.path.basename(file_path)}")
            else:
                messagebox.showwarning(
                    "Warning", "No file selected. The response will not be saved.")
                return  # Exit without saving if no file path is provided

        # Automatically save the updated responses to the file
        self.auto_save_responses()

    def auto_save_responses(self):
        # Gather the current data from the right table
        responses_data = {"responses": {}}
        for row in self.human_data_tree.get_children():
            query_id = self.human_data_tree.item(row, "values")[0]  # Query ID
            response_str = self.human_data_tree.item(
                row, "values")[2]  # Response (string)

            # If the response cell is empty, treat it as an empty list
            response_list = [item.strip() for item in response_str.split(
                ',') if item.strip()] if response_str.strip() else []

            responses_data["responses"][query_id] = response_list

        # Save the responses data to the file at the stored path
        with open(self.responses_file_path, 'w') as file:
            json.dump(responses_data, file, indent=4)

        # Print confirmation (optional)
        print(f"Auto-saved responses to {self.responses_file_path}")

    def load_responses_file(self):
        if not self.queries_loaded:  # Check if queries YAML has been loaded
            messagebox.showerror("Error", "Load a queries file first!")
            return

        file_path = filedialog.askopenfilename(
            title="Select a responses JSON file",
            filetypes=[("JSON files", "*.json")],
            initialdir=constants.RESPONSES_FOLDER_PATH
        )
        if file_path:
            try:
                # Attempt to load the JSON file
                with open(file_path, 'r') as file:
                    data = json.load(file)

                # Validate the structure of the JSON file
                if "responses" not in data or not isinstance(data["responses"], dict):
                    raise ValueError(
                        "The 'responses' key is missing or is not a dictionary.")

                # Store the path of the loaded responses file
                self.responses_file_path = file_path

                # Iterate over the table to match queries with responses from the JSON
                for row in self.human_data_tree.get_children():
                    query_id = self.human_data_tree.item(
                        row, "values")[0]  # Get the query ID from the table

                    # If the query ID exists in the responses JSON, update the Response column
                    if query_id in data["responses"]:
                        # Format the list of objects as a string
                        response_objects = ", ".join(
                            data["responses"][query_id])
                        self.human_data_tree.item(row, values=(
                            self.human_data_tree.item(
                                row)["values"][0],  # Query ID
                            self.human_data_tree.item(
                                row)["values"][1],  # Query text
                            response_objects))  # Response (list of objects)

                # Display the base name of the loaded responses file
                self.responses_label.config(
                    text=f"Loaded file: {os.path.basename(file_path)}")

            except (json.JSONDecodeError, ValueError) as e:
                # If any JSON or structural error occurs, display the specific error message
                messagebox.showerror(
                    "Error", f"Responses file is not in the desired format! Error: {str(e)}")

    def highlight_semantic_map_objects(self, event):
        # Clear existing highlights in the left table by setting all rows to default background color
        for row in self.semantic_map_tree.get_children():
            self.semantic_map_tree.item(row, tags=())  # Clear tags

        # Get the selected row in the right table
        selected_item = self.human_data_tree.focus()
        if not selected_item:
            return

        # Get the Response column value (list of object IDs as a string)
        response_str = self.human_data_tree.item(selected_item, "values")[2]

        # Parse the Response to get individual object IDs
        response_list = [obj_id.strip()
                         for obj_id in response_str.split(',') if obj_id.strip()]

        # Highlight matching rows in the left table
        for row in self.semantic_map_tree.get_children():
            object_id = self.semantic_map_tree.item(row, "values")[0]
            if object_id in response_list:
                self.semantic_map_tree.item(row, tags=("highlight",))

        # Configure the "highlight" tag style
        self.semantic_map_tree.tag_configure("highlight", background="yellow")

    def clear_queries_and_responses(self):
        # Clear all rows in the right table
        for row in self.human_data_tree.get_children():
            self.human_data_tree.delete(row)

        # Reset the loaded flags and file paths
        self.queries_loaded = False
        self.responses_file_path = None

        # Update the labels to indicate no files are loaded
        self.responses_label.config(text="No responses file loaded")

        # Optionally, clear any highlights in the left table as well
        for row in self.semantic_map_tree.get_children():
            self.semantic_map_tree.item(row, tags=())

        messagebox.showinfo(
            "Clear", "All queries and responses have been cleared.")

    def save_data(self):
        # Initialize the dictionary to store responses
        responses_data = {"responses": {}}

        # Iterate through the table to get Query ID and Response
        for row in self.human_data_tree.get_children():
            query_id = self.human_data_tree.item(row, "values")[0]  # Query ID
            response_str = self.human_data_tree.item(
                row, "values")[2]  # Response (string)

            # If the response cell is empty, treat it as an empty list
            if not response_str.strip():
                response_list = []
            else:
                # Convert the comma-separated response string into a list
                response_list = [item.strip()
                                 for item in response_str.split(',') if item.strip()]

            # Add the response list to the responses_data dictionary
            responses_data["responses"][query_id] = response_list

        # Check if a responses file path is already set
        if self.responses_file_path:
            # Save directly to the loaded responses file path
            file_path = self.responses_file_path
        else:
            # Open a file dialog to select where to save the JSON file
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json", filetypes=[("JSON files", "*.json")])
            if file_path:
                # Update the stored responses file path
                self.responses_file_path = file_path
            else:
                # If no path is selected, exit the function
                return

        # Save the responses data to the specified file as JSON
        with open(file_path, 'w') as file:
            json.dump(responses_data, file, indent=4)

        # Display a confirmation message after saving
        messagebox.showinfo(
            "Success", f"Responses successfully saved to {file_path}")


# Running the GUI
if __name__ == "__main__":
    app = GUI()
    app.mainloop()
