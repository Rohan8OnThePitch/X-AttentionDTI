import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Home from './pages/Home';
import Predict from './pages/Predict';
import NetworkBackground from './components/NetworkBackground'; // 1. IMPORT HERE
import { Activity } from 'lucide-react';

function App() {
  
  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <Router>
      {/* 2. ADD 'relative' TO THIS DIV */}
      <div className="min-h-screen flex flex-col text-gray-900 font-sans relative text-transparent-bg">
        
        {/* 3. DROP THE BACKGROUND HERE */}
        <NetworkBackground />

        {/* Professional Navigation Bar */}
        <header className="sticky top-0 z-50 bg-white/90 backdrop-blur-md shadow-sm border-b border-gray-200">
          <div className="max-w-6xl mx-auto px-4 py-4 flex justify-between items-center">
            
            <Link to="/" onClick={scrollToTop} className="flex items-center space-x-2 text-blue-600 hover:text-blue-800 transition">
              <Activity size={28} />
              <span className="text-xl font-bold tracking-tight">X-Attention DTI</span>
            </Link>
            
            <nav className="space-x-6">
              <Link to="/" onClick={scrollToTop} className="text-gray-600 hover:text-blue-600 font-medium">
                About Project
              </Link>
              
              <Link to="/predict" onClick={scrollToTop} className="bg-blue-600 text-white px-5 py-2 rounded-md font-medium hover:bg-blue-700 transition shadow-sm">
                Run Prediction
              </Link>
            </nav>
          </div>
        </header>

        {/* Main Content Area */}
        <main className="flex-grow z-10"> {/* Added z-10 so content stays above particles */}
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/predict" element={<Predict />} />
          </Routes>
        </main>

        {/* Minimal Footer */}
        <footer className="bg-gray-800 text-gray-400 py-6 text-center text-sm z-10"> {/* Added z-10 */}
          <p>© {new Date().getFullYear()} X-Attention DTI. Deep Learning for Drug Discovery.</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;