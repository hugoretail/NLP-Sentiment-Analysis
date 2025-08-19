# 📋 API Documentation

## 🎯 Analyse de Sentiment - API REST

Cette API permet d'analyser le sentiment de textes en français avec des scores de 1 à 5 étoiles.

---

## 🌐 Endpoints Disponibles

### 1. **Health Check**
```http
GET /health
```

**Réponse :**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "mode": "mock",
  "timestamp": "2025-08-19T15:30:00.000Z"
}
```

### 2. **Prédiction Simple**
```http
POST /predict
Content-Type: application/json

{
  "text": "Ce produit est fantastique!"
}
```

**Réponse :**
```json
{
  "success": true,
  "prediction": {
    "score": 4.2,
    "confidence": 0.85,
    "stars": 4,
    "label": "Positif",
    "timestamp": "2025-08-19T15:30:00.000Z"
  },
  "input_text": "Ce produit est fantastique!"
}
```

### 3. **Prédiction en Lot**
```http
POST /predict_batch
Content-Type: application/json

{
  "texts": [
    "Excellent service!",
    "Très décevant...",
    "Correct sans plus"
  ]
}
```

**Réponse :**
```json
{
  "success": true,
  "results": [
    {
      "index": 0,
      "text": "Excellent service!",
      "prediction": {
        "score": 4.5,
        "confidence": 0.92,
        "stars": 5,
        "label": "Très Positif"
      }
    },
    {
      "index": 1,
      "text": "Très décevant...",
      "prediction": {
        "score": 1.8,
        "confidence": 0.88,
        "stars": 2,
        "label": "Négatif"
      }
    }
  ],
  "total": 2
}
```

### 4. **Statistiques**
```http
GET /stats
```

**Réponse :**
```json
{
  "total_predictions": 156,
  "cache_hits": 23,
  "average_score": 3.2,
  "start_time": "2025-08-19T14:00:00.000Z",
  "cache_size": 45,
  "history_size": 100,
  "uptime_seconds": 5400
}
```

### 5. **Historique**
```http
GET /history
```

**Réponse :**
```json
{
  "history": [
    {
      "text": "Super application!",
      "result": {
        "score": 4.3,
        "label": "Positif",
        "timestamp": "2025-08-19T15:25:00.000Z"
      }
    }
  ],
  "total": 20
}
```

---

## 📊 Échelle de Sentiment

| Score | Étoiles | Label | Description |
|-------|---------|-------|-------------|
| 4.5 - 5.0 | ⭐⭐⭐⭐⭐ | Très Positif | Sentiment très positif |
| 3.5 - 4.4 | ⭐⭐⭐⭐☆ | Positif | Sentiment positif |
| 2.5 - 3.4 | ⭐⭐⭐☆☆ | Neutre | Sentiment neutre |
| 1.5 - 2.4 | ⭐⭐☆☆☆ | Négatif | Sentiment négatif |
| 1.0 - 1.4 | ⭐☆☆☆☆ | Très Négatif | Sentiment très négatif |

---

## ⚠️ Limites et Contraintes

- **Longueur maximale** : 512 caractères par texte
- **Batch maximum** : 100 textes par requête
- **Rate limiting** : Aucune limite en mode démo
- **Langue** : Optimisé pour le français
- **Mode** : Démonstration avec modèle simulé

---

## 🔧 Codes d'Erreur

| Code | Erreur | Description |
|------|--------|-------------|
| 400 | Bad Request | Données manquantes ou invalides |
| 413 | Payload Too Large | Texte ou batch trop volumineux |
| 500 | Internal Error | Erreur serveur interne |

**Exemple d'erreur :**
```json
{
  "error": "Texte trop long (max 512 caractères)",
  "success": false
}
```

---

## 💡 Exemples d'Utilisation

### Python
```python
import requests

# Prédiction simple
response = requests.post('http://localhost:5000/predict', 
    json={'text': 'Ce produit est génial!'})
result = response.json()
print(f"Score: {result['prediction']['score']}/5")
```

### JavaScript
```javascript
// Prédiction simple
const response = await fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text: 'Service excellent!'})
});
const data = await response.json();
console.log(`Sentiment: ${data.prediction.label}`);
```

### cURL
```bash
# Test de l'API
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Application formidable!"}'
```

---

## 🚀 Démarrage Rapide

1. **Installation :**
   ```bash
   pip install flask flask-cors requests
   ```

2. **Démarrage :**
   ```bash
   python start.py
   ```

3. **Test :**
   ```bash
   curl http://localhost:5000/health
   ```

4. **Interface :**
   - Dashboard : http://localhost:5000
   - Simple : http://localhost:5000/simple

---

*Version 1.0 - Mode Démonstration*
