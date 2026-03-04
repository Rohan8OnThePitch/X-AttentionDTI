import torch
torch.set_num_threads(1) # Limits CPU usage to prevent memory spikes
import math
from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS

from models.model import DTIModel
from preprocessing.drug_preprocessing import build_drug_tensors
from preprocessing.protein_preprocessing import build_protein_tensors

app = Flask(__name__)
# Enable CORS for all routes so React can connect securely
CORS(app) 

device = torch.device("cpu")

print("Loading trained X-Attention DTI model...")

model = DTIModel(hidden_dim=512)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.to(device)
model.eval()
# Add this line to delete unnecessary weights after loading
with torch.no_grad():
    pass

print("Model loaded successfully.")

# Change the route to /api/predict and only allow POST requests
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from React instead of an HTML form
        data = request.get_json()
        smiles = data.get("smiles")
        protein = data.get("protein")

        if not smiles or not protein:
             return jsonify({"error": "Missing SMILES or Protein sequence"}), 400

        # Drug tensors
        node_features, hyperedge_indices, hyperedge_types, batch_indices = build_drug_tensors(smiles)

        # Protein tensors
        input_ids, attention_mask, teacher_cls = build_protein_tensors(protein)

        # Move to device
        node_features = node_features.to(device)
        hyperedge_indices = hyperedge_indices.to(device)
        hyperedge_types = hyperedge_types.to(device)
        batch_indices = batch_indices.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        teacher_cls = teacher_cls.to(device)

        with torch.no_grad():
            output = model(
                node_features=node_features,
                hyperedge_indices=hyperedge_indices,
                hyperedge_types=hyperedge_types,
                batch_indices=batch_indices,
                protein_input_ids=input_ids,
                protein_attention_mask=attention_mask,
                teacher_cls=teacher_cls,
            )

        log_kiba = output["pred_affinity"].item()

        # Convert back to original KIBA scale
        kiba_score = math.exp(log_kiba) - 1

        # Return JSON data to React
        return jsonify({"prediction": round(kiba_score, 4)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Specify port 5000 explicitly
    app.run(debug=True, port=5000)