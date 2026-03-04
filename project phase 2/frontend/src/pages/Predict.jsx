import { useState } from 'react';
import axios from 'axios';
import { Activity, AlertCircle, CheckCircle2, RotateCcw, FlaskConical } from 'lucide-react';

export default function Predict() {
  const [smiles, setSmiles] = useState('');
  const [protein, setProtein] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post('http://127.0.0.1:5000/api/predict', {
        smiles,
        protein
      });
      
      setResult(response.data.prediction);
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred during prediction.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSmiles('');
    setProtein('');
    setResult(null);
    setError(null);
  };

  // Function to load sample data (Imatinib + standard target)
  const handleLoadExample = () => {
    setSmiles('CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5');
    setProtein('MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR');
    setResult(null);
    setError(null);
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-12">
      <div className="bg-white/95 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-100 p-8">
        
        {/* Header Row with Example Button */}
        <div className="flex flex-col md:flex-row md:items-center justify-between mb-6 border-b border-gray-100 pb-4">
          <h2 className="text-3xl font-bold text-gray-900 flex items-center mb-4 md:mb-0">
            <Activity className="mr-3 text-blue-600" />
            Run DTI Prediction
          </h2>
          <button
            type="button"
            onClick={handleLoadExample}
            className="flex items-center text-sm font-medium text-blue-600 bg-blue-50 hover:bg-blue-100 px-4 py-2 rounded-lg transition"
          >
            <FlaskConical size={16} className="mr-2" />
            Load Example Data
          </button>
        </div>
        
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Drug (SMILES String)
            </label>
            <textarea
              value={smiles}
              onChange={(e) => setSmiles(e.target.value)}
              placeholder="e.g., CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"
              className="w-full p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 font-mono text-sm shadow-sm"
              rows="3"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Target Protein (Amino Acid Sequence)
            </label>
            <textarea
              value={protein}
              onChange={(e) => setProtein(e.target.value)}
              placeholder="e.g., MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
              className="w-full p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 font-mono text-sm shadow-sm"
              rows="6"
              required
            />
          </div>

          <div className="flex space-x-4">
            <button
              type="button"
              onClick={handleClear}
              disabled={loading || (!smiles && !protein && !result)}
              className="w-1/4 bg-gray-100 text-gray-700 font-bold py-4 rounded-lg hover:bg-gray-200 transition disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center shadow-sm"
            >
              <RotateCcw className="mr-2" size={20} />
              Clear
            </button>

            <button
              type="submit"
              disabled={loading || !smiles || !protein}
              className="w-3/4 bg-blue-600 text-white font-bold py-4 rounded-lg hover:bg-blue-700 transition disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center shadow-sm"
            >
              {loading ? 'Analyzing Binding Affinity...' : 'Predict Affinity (KIBA)'}
            </button>
          </div>
        </form>

        {error && (
          <div className="mt-8 p-4 bg-red-50 border border-red-200 text-red-700 rounded-lg flex items-start animate-fade-in">
            <AlertCircle className="mr-3 shrink-0 mt-0.5" size={20} />
            <p>{error}</p>
          </div>
        )}

        {result !== null && (
          <div className="mt-8 p-6 bg-green-50 border border-green-200 rounded-lg flex flex-col items-center justify-center text-center shadow-inner animate-fade-in">
            <CheckCircle2 className="text-green-600 mb-2" size={32} />
            <h3 className="text-lg font-medium text-green-900 mb-1">Predicted Binding Affinity</h3>
            <p className="text-4xl font-extrabold text-green-700">{result}</p>
          </div>
        )}
      </div>
    </div>
  );
}