from flask import Flask, request, render_template
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

app = Flask(__name__)

# =========================
# LOAD DATASET
# =========================
dataset = Planetoid(root='./data', name='CiteSeer')
data = dataset[0]

class_names = {
    0: "Agents",
    1: "AI",
    2: "DB",
    3: "IR",
    4: "ML",
    5: "HCI"
}

# =========================
# GCN MODEL
# =========================
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# =========================
# TRAIN MODEL
# =========================
model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(50):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data).argmax(dim=1)

# =========================
# HOME PAGE
# =========================
@app.route('/')
def home():
    return render_template('index.html')

# =========================
# PREDICT NODE CLASS
# =========================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        node_id = int(request.form['node_id'])

        if node_id < 0 or node_id >= data.num_nodes:
            return render_template(
                'result.html',
                node_id=node_id,
                predicted_class="Invalid Node ID",
                actual_class="N/A",
                neighbors=[],
                reason="Please enter a valid node ID within dataset range."
            )

        predicted_class = class_names[pred[node_id].item()]
        actual_class = class_names[data.y[node_id].item()]

        neighbors = data.edge_index[1][data.edge_index[0] == node_id].tolist()
        neighbor_info = []

        for n in neighbors[:5]:
            neighbor_info.append({
                "id": n,
                "class": class_names[data.y[n].item()]
            })

        reason = (
            f"Node {node_id} is predicted as '{predicted_class}' because it is connected "
            f"to neighboring papers with related citation patterns and similar feature representations."
        )

        return render_template(
            'result.html',
            node_id=node_id,
            predicted_class=predicted_class,
            actual_class=actual_class,
            neighbors=neighbor_info,
            reason=reason
        )

    except:
        return render_template(
            'result.html',
            node_id="Invalid",
            predicted_class="Error",
            actual_class="N/A",
            neighbors=[],
            reason="Please enter a valid numeric Node ID."
        )

# =========================
# RUN APP
# =========================
if __name__ == '__main__':
    app.run(debug=True)
