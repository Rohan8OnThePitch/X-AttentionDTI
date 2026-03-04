import { ArrowRight, Dna, Network, Zap } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function Home() {
  return (
    <div className="max-w-6xl mx-auto px-4 py-12">
      <section className="text-center py-16 lg:py-24">
        <h1 className="text-4xl md:text-6xl font-extrabold text-gray-900 mb-6 tracking-tight">
          Next-Generation <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-cyan-500">Drug-Target Interaction</span>
        </h1>
        <p className="text-xl text-gray-600 mb-10 max-w-3xl mx-auto leading-relaxed">
          X-Attention DTI leverages advanced hypergraph neural networks and protein language models (ESM-2) to predict binding affinities with unprecedented accuracy.
        </p>
        <Link to="/predict" className="inline-flex items-center space-x-2 bg-blue-600 text-white px-8 py-4 rounded-lg text-lg font-semibold hover:bg-blue-700 transition shadow-lg">
          <span>Start Predicting</span>
          <ArrowRight size={20} />
        </Link>
      </section>

      <hr className="border-gray-200 my-12" />

      <section className="py-12">
        <h2 className="text-3xl font-bold text-center mb-12">Under the Hood</h2>
        <div className="grid md:grid-cols-3 gap-8">
          <div className="bg-white p-8 rounded-xl shadow-sm border border-gray-100">
            <div className="w-12 h-12 bg-blue-100 text-blue-600 rounded-lg flex items-center justify-center mb-6">
              <Network size={28} />
            </div>
            <h3 className="text-xl font-bold mb-3">Hypergraph Drug Encoder</h3>
            <p className="text-gray-600 leading-relaxed">
              Captures complex multi-way atomic interactions and ring structures using degree-normalized hypergraph convolutions.
            </p>
          </div>

          <div className="bg-white p-8 rounded-xl shadow-sm border border-gray-100">
            <div className="w-12 h-12 bg-purple-100 text-purple-600 rounded-lg flex items-center justify-center mb-6">
              <Dna size={28} />
            </div>
            <h3 className="text-xl font-bold mb-3">ESM-2 Protein Modeling</h3>
            <p className="text-gray-600 leading-relaxed">
              Utilizes Facebook's state-of-the-art evolutionary scale modeling. We fuse representations from student and teacher models.
            </p>
          </div>

          <div className="bg-white p-8 rounded-xl shadow-sm border border-gray-100">
            <div className="w-12 h-12 bg-teal-100 text-teal-600 rounded-lg flex items-center justify-center mb-6">
              <Zap size={28} />
            </div>
            <h3 className="text-xl font-bold mb-3">Cross-Attention</h3>
            <p className="text-gray-600 leading-relaxed">
              Dynamically aligns the sequence-level protein embeddings with the hypergraph node-level drug embeddings.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}