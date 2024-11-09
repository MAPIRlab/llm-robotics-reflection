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

        # Initialization
        self.semantic_map_loaded = False
        self.queries_loaded = False
        self.responses_file_path = None

        self.title("Semantic Map and Human Data Interface")
        self.geometry("1200x800")

        # Paned Window Setup
        paned_window = tk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)

        # Left Frame (Semantic Map)
        left_frame = tk.Frame(paned_window, bg='lightblue')
        paned_window.add(left_frame)
        tk.Label(left_frame, text="Semantic map",
                 font=("Arial", 12, "bold")).pack(pady=10)
        tk.Button(left_frame, text="Load semantic map JSON file",
                  command=self.load_semantic_map).pack(pady=5)
        self.semantic_map_label = tk.Label(
            left_frame, text="No semantic map file loaded", font=("Arial", 10, "italic"))
        self.semantic_map_label.pack(pady=5)

        self.semantic_map_tree = ttk.Treeview(
            left_frame,
            columns=("Object id", "Most Probable Object",
                     "BoundingBoxCenter", "BoundingBoxSize", "Results"),
            show='headings',
            height=10
        )
        for col in self.semantic_map_tree["columns"]:
            self.semantic_map_tree.heading(col, text=col)
        self.semantic_map_tree.pack(
            padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Right Frame (Human Data)
        right_frame = tk.Frame(paned_window, bg='lightyellow')
        paned_window.add(right_frame)
        tk.Label(right_frame, text="Human data", font=(
            "Arial", 12, "bold")).pack(pady=10)
        tk.Button(right_frame, text="Load query YAML file",
                  command=self.load_query_file).pack(pady=5)
        tk.Button(right_frame, text="Load responses JSON file",
                  command=self.load_responses_file).pack(pady=5)
        self.responses_label = tk.Label(
            right_frame, text="No responses file loaded", font=("Arial", 10, "italic"))
        self.responses_label.pack(pady=5)

        self.human_data_tree = ttk.Treeview(
            right_frame,
            columns=("Query ID", "Query", "Response"),
            show='headings',
            height=10
        )
        for col in self.human_data_tree["columns"]:
            self.human_data_tree.heading(col, text=col)
        self.human_data_tree.column("Response", width=200)
        self.human_data_tree.bind("<Double-1>", self.on_double_click)
        self.human_data_tree.bind(
            "<<TreeviewSelect>>", self.highlight_semantic_map_objects)
        self.human_data_tree.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        tk.Button(right_frame, text="Clear", command=self.clear_queries_and_responses,
                  bg="red", fg="white").pack(pady=5)

    def clear_treeview(self, treeview):
        treeview.delete(*treeview.get_children())

    def load_semantic_map(self):
        file_path = filedialog.askopenfilename(
            title="Select a semantic map JSON file",
            filetypes=[("JSON files", "*.json")],
            initialdir=constants.SEMANTIC_MAPS_FOLDER_PATH
        )
        if file_path:
            self.clear_queries_and_responses()
            try:
                with open(file_path, 'r') as file:
                    semantic_map_obj = json.load(file)
                semantic_map_obj = preprocess.preprocess_semantic_map(
                    semantic_map_obj, True)
                instances = semantic_map_obj.get("instances", {})
                if not isinstance(instances, dict):
                    raise ValueError("Invalid 'instances' structure.")

                self.clear_treeview(self.semantic_map_tree)
                sorted_obj_ids = sorted(
                    instances.keys(),
                    key=lambda x: [int(t) if t.isdigit() else t.lower()
                                   for t in re.split(r'(\d+)', x)]
                )

                for obj_id in sorted_obj_ids:
                    obj_data = instances[obj_id]
                    bbox = obj_data.get("bbox", {})
                    results = obj_data.get("results", {})
                    if "center" not in bbox or not results:
                        raise ValueError(
                            f"Invalid structure for object '{obj_id}'.")

                    bbox_center = tuple(bbox["center"])
                    bbox_size = tuple(bbox.get("size", []))
                    sorted_results = sorted(
                        results.items(), key=lambda x: x[1], reverse=True)
                    total_score = sum(score for _, score in sorted_results)
                    if total_score > 0:
                        label, score = sorted_results[0]
                        most_probable = f"{label} ({score / total_score:.2f})"
                    else:
                        most_probable = "unknown (0.00)"
                    results_str = ", ".join(
                        f"{k}: {v:.2f}" for k, v in sorted_results)

                    self.semantic_map_tree.insert("", "end", values=(
                        obj_id, most_probable, bbox_center, bbox_size, results_str))

                self.semantic_map_loaded = True
                self.semantic_map_label.config(
                    text=f"Loaded file: {os.path.basename(file_path)}")
            except (json.JSONDecodeError, ValueError) as e:
                messagebox.showerror(
                    "Error", f"Invalid semantic map format! Error: {e}")

    def load_query_file(self):
        if not self.semantic_map_loaded:
            messagebox.showerror("Error", "Load a semantic map file first!")
            return
        file_path = filedialog.askopenfilename(
            title="Select a query YAML file",
            filetypes=[("YAML files", "*.yaml")],
            initialdir=constants.DATA_FOLDER_PATH
        )
        if file_path:
            try:
                data = file_utils.load_yaml(file_path)
                queries = data.get("queries", {})
                if not isinstance(queries, dict):
                    raise ValueError("Invalid 'queries' structure.")

                self.clear_treeview(self.human_data_tree)
                for query_id, query_text in queries.items():
                    self.human_data_tree.insert(
                        "", "end", values=(query_id, query_text, ""))
                self.queries_loaded = True
            except (yaml.YAMLError, ValueError) as e:
                messagebox.showerror(
                    "Error", f"Invalid queries file format! Error: {e}")

    def on_double_click(self, event):
        item_id = self.human_data_tree.focus()
        column = self.human_data_tree.identify_column(event.x)
        if column == '#3':
            bbox = self.human_data_tree.bbox(item_id, column)
            if bbox:
                current_value = self.human_data_tree.item(item_id, 'values')[2]
                if hasattr(self, 'current_entry_widget'):
                    self.current_entry_widget.destroy()
                self.current_entry_widget = tk.Entry(self.human_data_tree)
                self.current_entry_widget.insert(0, current_value)
                self.current_entry_widget.bind(
                    "<Return>",
                    lambda e: self.save_response(
                        item_id, self.current_entry_widget.get())
                )
                self.current_entry_widget.place(
                    x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3])
                self.current_entry_widget.focus()

    def save_response(self, item_id, new_value):
        response_list = [item.strip()
                         for item in new_value.split(',') if item.strip()]
        existing_ids = {self.semantic_map_tree.item(
            row, "values")[0] for row in self.semantic_map_tree.get_children()}
        if len(response_list) != len(set(response_list)):
            messagebox.showerror("Error", "Duplicate object IDs in response.")
            return
        if not set(response_list).issubset(existing_ids):
            messagebox.showerror(
                "Error", "Some object IDs do not exist in the semantic map.")
            return

        values = self.human_data_tree.item(item_id, "values")
        self.human_data_tree.item(item_id, values=(
            values[0], values[1], ", ".join(response_list)))
        if hasattr(self, 'current_entry_widget'):
            self.current_entry_widget.destroy()
        self.highlight_semantic_map_objects(None)

        if not self.responses_file_path:
            file_path = filedialog.asksaveasfilename(
                title="Save Responses File",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")]
            )
            if file_path:
                self.responses_file_path = file_path
                self.responses_label.config(
                    text=f"Loaded file: {os.path.basename(file_path)}")
            else:
                messagebox.showwarning(
                    "Warning", "Response not saved. No file selected.")
                return
        self.auto_save_responses()

    def auto_save_responses(self):
        responses_data = {"responses": {}}
        for row in self.human_data_tree.get_children():
            query_id, _, response_str = self.human_data_tree.item(
                row, "values")
            response_list = [item.strip()
                             for item in response_str.split(',') if item.strip()]
            responses_data["responses"][query_id] = response_list
        with open(self.responses_file_path, 'w') as file:
            json.dump(responses_data, file, indent=4)
        print(f"Auto-saved responses to {self.responses_file_path}")

    def load_responses_file(self):
        if not self.queries_loaded:
            messagebox.showerror("Error", "Load a queries file first!")
            return
        file_path = filedialog.askopenfilename(
            title="Select a responses JSON file",
            filetypes=[("JSON files", "*.json")],
            initialdir=constants.RESPONSES_FOLDER_PATH
        )
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                responses = data.get("responses", {})
                if not isinstance(responses, dict):
                    raise ValueError("Invalid 'responses' structure.")

                self.responses_file_path = file_path
                for row in self.human_data_tree.get_children():
                    query_id = self.human_data_tree.item(row, "values")[0]
                    response_objects = ", ".join(responses.get(query_id, []))
                    values = self.human_data_tree.item(row, "values")
                    self.human_data_tree.item(row, values=(
                        values[0], values[1], response_objects))
                self.responses_label.config(
                    text=f"Loaded file: {os.path.basename(file_path)}")
            except (json.JSONDecodeError, ValueError) as e:
                messagebox.showerror(
                    "Error", f"Invalid responses file format! Error: {e}")

    def highlight_semantic_map_objects(self, event):
        self.semantic_map_tree.tag_configure("highlight", background="yellow")
        for row in self.semantic_map_tree.get_children():
            self.semantic_map_tree.item(row, tags=())
        selected_item = self.human_data_tree.focus()
        if selected_item:
            response_str = self.human_data_tree.item(
                selected_item, "values")[2]
            response_list = [obj_id.strip()
                             for obj_id in response_str.split(',') if obj_id.strip()]
            for row in self.semantic_map_tree.get_children():
                object_id = self.semantic_map_tree.item(row, "values")[0]
                if object_id in response_list:
                    self.semantic_map_tree.item(row, tags=("highlight",))

    def clear_queries_and_responses(self):
        self.clear_treeview(self.human_data_tree)
        self.queries_loaded = False
        self.responses_file_path = None
        self.responses_label.config(text="No responses file loaded")
        for row in self.semantic_map_tree.get_children():
            self.semantic_map_tree.item(row, tags=())
        messagebox.showinfo(
            "Clear", "All queries and responses have been cleared.")


if __name__ == "__main__":
    app = GUI()
    app.mainloop()
