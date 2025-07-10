
import React from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  Download, 
  ArrowRight, 
  Mail, 
  MessageSquare, 
  BarChart3, 
  Settings, 
  Database,
  Workflow,
  FileSpreadsheet,
  Zap,
  TrendingUp,
  Users,
  CheckCircle,
  ExternalLink,
  Star,
  Calendar,
  Phone
} from 'lucide-react';
import Hero from '@/components/Hero';
import About from '@/components/About';
import Skills from '@/components/Skills';
import Projects from '@/components/Projects';
import Testimonials from '@/components/Testimonials';
import Blog from '@/components/Blog';
import Downloads from '@/components/Downloads';
import Contact from '@/components/Contact';

const Index = () => {
  return (
    <div className="min-h-screen bg-white">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 bg-white/95 backdrop-blur-sm border-b border-gray-200 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="font-bold text-xl text-gray-900">
              DS Portfolio
            </div>
            <div className="hidden md:flex space-x-8">
              <a href="#home" className="text-gray-600 hover:text-blue-600 transition-colors">Home</a>
              <a href="#about" className="text-gray-600 hover:text-blue-600 transition-colors">About</a>
              <a href="#skills" className="text-gray-600 hover:text-blue-600 transition-colors">Skills</a>
              <a href="#projects" className="text-gray-600 hover:text-blue-600 transition-colors">Projects</a>
              <a href="#blog" className="text-gray-600 hover:text-blue-600 transition-colors">Blog</a>
              <a href="#contact" className="text-gray-600 hover:text-blue-600 transition-colors">Contact</a>
            </div>
            <Button className="bg-blue-600 hover:bg-blue-700">
              Let's Automate Together
            </Button>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main>
        <Hero />
        <About />
        <Skills />
        <Projects />
        <Testimonials />
        <Blog />
        <Downloads />
        <Contact />
      </main>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <h3 className="font-bold text-lg mb-4">Data Scientist & Workflow Automation Specialist</h3>
              <p className="text-gray-400">
                Transforming business processes through intelligent automation and data-driven insights.
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Quick Links</h4>
              <div className="space-y-2">
                <a href="#about" className="block text-gray-400 hover:text-white transition-colors">About</a>
                <a href="#projects" className="block text-gray-400 hover:text-white transition-colors">Projects</a>
                <a href="#blog" className="block text-gray-400 hover:text-white transition-colors">Blog</a>
                <a href="#contact" className="block text-gray-400 hover:text-white transition-colors">Contact</a>
              </div>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Get In Touch</h4>
              <div className="space-y-2">
                <a href="mailto:contact@example.com" className="flex items-center text-gray-400 hover:text-white transition-colors">
                  <Mail className="h-4 w-4 mr-2" />
                  contact@example.com
                </a>
                <a href="#" className="flex items-center text-gray-400 hover:text-white transition-colors">
                  <MessageSquare className="h-4 w-4 mr-2" />
                  WhatsApp Consultation
                </a>
              </div>
            </div>
          </div>
          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400">
            <p>&copy; 2024 Data Scientist & Workflow Automation Specialist. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
