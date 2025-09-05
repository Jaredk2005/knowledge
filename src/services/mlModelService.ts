export interface MLPredictionRequest {
  source_ip: string;
  destination_ip?: string;
  packet_size?: number;
  protocol?: string;
  payload?: string;
  timestamp?: string;
  additional_features?: Record<string, any>;
}

export interface MLPredictionResponse {
  threat_type: string;
  confidence: number;
  severity: number;
  is_threat: boolean;
  classification_method: string;
  anomaly_detected: boolean;
  rules_triggered: string[];
  processing_time_ms: number;
}

export interface ModelInfo {
  model_status: string;
  feature_count: number;
  threat_categories: string[];
  last_trained?: string;
  accuracy?: number;
  f1_score?: number;
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  auc_score: number;
  training_samples: number;
  test_samples: number;
  training_time: number;
  last_trained: string;
  feature_importance: Record<string, number>;
  confusion_matrix: number[][];
}

class MLModelService {
  private baseUrl = 'http://localhost:8003/api/ml';

  async predictThreat(request: MLPredictionRequest): Promise<MLPredictionResponse> {
    const response = await fetch(`${this.baseUrl}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Prediction failed: ${response.statusText}`);
    }

    return await response.json();
  }

  async predictThreatsBatch(requests: MLPredictionRequest[]): Promise<any> {
    const response = await fetch(`${this.baseUrl}/predict/batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requests),
    });

    if (!response.ok) {
      throw new Error(`Batch prediction failed: ${response.statusText}`);
    }

    return await response.json();
  }

  async getModelInfo(): Promise<ModelInfo> {
    const response = await fetch(`${this.baseUrl}/model/info`);
    
    if (!response.ok) {
      throw new Error(`Failed to get model info: ${response.statusText}`);
    }

    return await response.json();
  }

  async getModelMetrics(): Promise<ModelMetrics> {
    const response = await fetch(`${this.baseUrl}/model/metrics`);
    
    if (!response.ok) {
      throw new Error(`Failed to get model metrics: ${response.statusText}`);
    }

    return await response.json();
  }

  async trainModel(config: {
    use_database?: boolean;
    augment_data?: boolean;
    training_samples?: number;
  }): Promise<any> {
    const response = await fetch(`${this.baseUrl}/model/train`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config),
    });

    if (!response.ok) {
      throw new Error(`Training failed: ${response.statusText}`);
    }

    return await response.json();
  }

  async trainModelWithCSV(config: {
    training_data: any[];
    label_column: string;
    feature_columns: string[];
    use_csv: boolean;
  }): Promise<any> {
    const response = await fetch(`${this.baseUrl}/model/train-csv`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config),
    });

    if (!response.ok) {
      throw new Error(`CSV training failed: ${response.statusText}`);
    }

    return await response.json();
  }
  async getModelReport(): Promise<{ report: string }> {
    const response = await fetch(`${this.baseUrl}/model/report`);
    
    if (!response.ok) {
      throw new Error(`Failed to get model report: ${response.statusText}`);
    }

    return await response.json();
  }

  async benchmarkModel(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/model/benchmark`, {
      method: 'POST'
    });

    if (!response.ok) {
      throw new Error(`Benchmark failed: ${response.statusText}`);
    }

    return await response.json();
  }

  async healthCheck(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/health`);
    
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }

    return await response.json();
  }
}

export const mlModelService = new MLModelService();