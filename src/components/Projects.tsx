
import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { ExternalLink, TrendingUp, Users, Clock, CheckCircle, ArrowRight, Github, Eye, Code, Database, Brain, LineChart } from 'lucide-react';

const Projects = () => {
  const [selectedProject, setSelectedProject] = useState(null);

  const projects = [
    {
      title: "Real-Time Customer Churn Prediction System",
      description: "End-to-end ML pipeline predicting customer churn for Safaricom Kenya using ensemble methods, real-time feature engineering, and automated model retraining. Deployed on AWS with 94% accuracy.",
      image: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=600&h=400&fit=crop",
      tags: ["Python", "Scikit-learn", "AWS", "Docker", "Apache Airflow", "PostgreSQL"],
      metrics: [
        { label: "Model Accuracy", value: "94.2%", icon: CheckCircle },
        { label: "Churn Reduction", value: "23%", icon: TrendingUp },
        { label: "Customers Analyzed", value: "2.1M", icon: Users }
      ],
      category: "Machine Learning",
      detailedDescription: "Built a comprehensive customer churn prediction system for Safaricom Kenya, processing over 2.1 million customer records. The system combines traditional ML algorithms with advanced feature engineering to identify at-risk customers 30 days before churn.",
      keyFeatures: [
        "Real-time data ingestion from multiple sources (billing, usage, support tickets)",
        "Advanced feature engineering including RFM analysis and behavioral patterns",
        "Ensemble model combining Random Forest, XGBoost, and Neural Networks",
        "Automated model retraining pipeline with drift detection",
        "A/B testing framework for retention campaign optimization",
        "Interactive Streamlit dashboard for business stakeholders",
        "API endpoints for real-time scoring and batch predictions"
      ],
      technologies: ["Python", "Pandas", "Scikit-learn", "XGBoost", "TensorFlow", "Apache Airflow", "Docker", "AWS SageMaker", "PostgreSQL", "Redis", "Streamlit"],
      implementation: {
        dataCollection: "Integrated data from CRM, billing systems, network usage logs, and customer service interactions using Apache Kafka for real-time streaming.",
        featureEngineering: "Created 150+ features including customer lifetime value, usage patterns, payment behavior, and social network analysis.",
        modelDevelopment: "Implemented ensemble approach with hyperparameter tuning using Optuna. Cross-validation score: 0.942 AUC.",
        deployment: "Containerized using Docker, deployed on AWS ECS with auto-scaling based on prediction volume.",
        monitoring: "Real-time model performance monitoring with alerts for data drift and prediction accuracy degradation."
      },
      results: [
        "Reduced customer churn by 23% through targeted retention campaigns",
        "Identified $2.4M in potential revenue at risk monthly",
        "Improved marketing campaign ROI by 45%",
        "Reduced false positive rate to 8.3% from previous 23%"
      ],
      github: "https://github.com/henrymuli/churn-prediction-system",
      demo: "https://churn-predictor-demo.streamlit.app",
      documentation: "Complete documentation with setup instructions, API reference, and model explanations"
    },
    {
      title: "Agricultural Yield Optimization Using Satellite Imagery",
      description: "Computer vision system analyzing satellite imagery and IoT sensor data to predict crop yields and optimize farming decisions for 5,000+ Kenyan smallholder farmers.",
      image: "https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?q=80&w=600&h=400&fit=crop",
      tags: ["Python", "TensorFlow", "Google Earth Engine", "Computer Vision", "IoT"],
      metrics: [
        { label: "Yield Increase", value: "28%", icon: TrendingUp },
        { label: "Farmers Helped", value: "5,247", icon: Users },
        { label: "Prediction Accuracy", value: "91.5%", icon: CheckCircle }
      ],
      category: "Computer Vision",
      detailedDescription: "Developed an AI-powered agricultural advisory system combining satellite imagery analysis, weather forecasting, and IoT sensor data to provide personalized recommendations to smallholder farmers across Kenya.",
      keyFeatures: [
        "Multi-spectral satellite image analysis using CNNs for crop health assessment",
        "Weather pattern recognition and 14-day yield forecasting",
        "Soil moisture prediction using time series analysis",
        "Mobile app with offline capability for rural areas",
        "Integration with local weather stations and IoT sensors",
        "Multi-language support (English, Swahili, Kikuyu)",
        "SMS-based recommendations for farmers without smartphones"
      ],
      technologies: ["Python", "TensorFlow", "Keras", "Google Earth Engine", "OpenCV", "React Native", "FastAPI", "MongoDB", "Twilio API"],
      implementation: {
        dataCollection: "Processed 50TB+ of satellite imagery from Sentinel-2 and Landsat missions. Integrated weather data from 200+ stations across Kenya.",
        modelDevelopment: "Built U-Net architecture for crop segmentation and LSTM networks for yield prediction. Achieved 91.5% accuracy on test farms.",
        mobileApp: "React Native app with offline sync, GPS mapping, and camera integration for field photos.",
        deployment: "Deployed on Google Cloud Platform with auto-scaling. API handles 10,000+ requests daily.",
        partnerships: "Collaborated with Kenya Agricultural Research Institute (KALRO) and local cooperatives."
      },
      results: [
        "Average yield increase of 28% for participating farmers",
        "Reduced water usage by 35% through precision irrigation recommendations",
        "Prevented crop losses worth $1.2M through early disease detection",
        "Expanded to 3 counties with plans for nationwide rollout"
      ],
      github: "https://github.com/henrymuli/agri-yield-prediction",
      demo: "https://crop-advisor-kenya.herokuapp.com",
      documentation: "Includes mobile app setup, satellite data processing pipeline, and farmer onboarding guides"
    },
    {
      title: "Real-Time Fraud Detection for Mobile Money",
      description: "Advanced anomaly detection system for M-Pesa transactions using unsupervised learning and graph neural networks, processing 1M+ daily transactions with sub-100ms latency.",
      image: "https://images.unsplash.com/photo-1563013544-824ae1b704d3?q=80&w=600&h=400&fit=crop",
      tags: ["Python", "PyTorch", "Apache Kafka", "Redis", "Kubernetes", "Graph ML"],
      metrics: [
        { label: "Fraud Reduction", value: "78%", icon: CheckCircle },
        { label: "Processing Speed", value: "<50ms", icon: Clock },
        { label: "Daily Transactions", value: "1.2M+", icon: TrendingUp }
      ],
      category: "Anomaly Detection",
      detailedDescription: "Built a sophisticated real-time fraud detection system processing over 1.2 million M-Pesa transactions daily. The system uses graph neural networks to detect fraud rings and advanced anomaly detection for suspicious patterns.",
      keyFeatures: [
        "Real-time transaction scoring using Isolation Forest and autoencoders",
        "Graph neural network analysis to detect coordinated fraud attacks",
        "Dynamic risk scoring with contextual factors (time, location, amount)",
        "Automated rule engine with ML-based threshold optimization",
        "Real-time alerts and case management system",
        "Integration with existing fraud investigation workflows",
        "Explainable AI features for regulatory compliance"
      ],
      technologies: ["Python", "PyTorch", "Scikit-learn", "Apache Kafka", "Redis", "PostgreSQL", "Docker", "Kubernetes", "Grafana", "ELK Stack"],
      implementation: {
        architecture: "Microservices architecture with event-driven processing. Kafka streams handle real-time data flow.",
        modelDevelopment: "Ensemble of Isolation Forest, One-Class SVM, and Graph Attention Networks. Weekly model retraining with active learning.",
        scalability: "Kubernetes deployment with horizontal pod autoscaling. Handles traffic spikes during peak hours.",
        monitoring: "Real-time dashboards showing transaction volumes, fraud rates, and model performance metrics.",
        compliance: "GDPR and PCI DSS compliant with audit trails and explainable decisions."
      },
      results: [
        "Reduced fraudulent transactions by 78% within 6 months",
        "Prevented financial losses of $3.2M annually",
        "Improved customer trust and reduced complaints by 45%",
        "False positive rate reduced to 2.1% from previous 12%"
      ],
      github: "https://github.com/henrymuli/mpesa-fraud-detection",
      demo: "https://fraud-detection-dashboard.herokuapp.com",
      documentation: "Complete system architecture, model explanations, and deployment guides"
    },
    {
      title: "Healthcare Resource Optimization Platform",
      description: "Predictive analytics platform for Kenyatta National Hospital optimizing patient flow, staff scheduling, and resource allocation using time series forecasting and optimization algorithms.",
      image: "https://images.unsplash.com/photo-1576091160399-112ba8d25d1f?q=80&w=600&h=400&fit=crop",
      tags: ["Python", "Prophet", "Optimization", "Streamlit", "PostgreSQL"],
      metrics: [
        { label: "Wait Time Reduction", value: "47%", icon: Clock },
        { label: "Resource Efficiency", value: "62%", icon: TrendingUp },
        { label: "Patients Served", value: "15K+", icon: Users }
      ],
      category: "Healthcare Analytics",
      detailedDescription: "Developed a comprehensive healthcare management system for Kenya's largest referral hospital, processing patient data, staff schedules, and resource utilization to optimize hospital operations and improve patient outcomes.",
      keyFeatures: [
        "Patient admission forecasting using Facebook Prophet and SARIMA models",
        "Staff scheduling optimization using linear programming",
        "Real-time bed occupancy tracking and allocation algorithms",
        "Emergency department flow optimization",
        "Predictive maintenance for medical equipment",
        "Interactive dashboards for different hospital departments",
        "Integration with existing Hospital Management System"
      ],
      technologies: ["Python", "Prophet", "Scikit-learn", "PuLP", "Streamlit", "PostgreSQL", "Plotly", "Apache Superset", "FastAPI"],
      implementation: {
        dataIntegration: "Connected to hospital's HMIS system, processing 50,000+ patient records monthly.",
        forecasting: "Multi-seasonal forecasting models accounting for holidays, disease outbreaks, and weather patterns.",
        optimization: "Linear programming models for staff scheduling with constraint satisfaction.",
        visualization: "Real-time dashboards showing KPIs, resource utilization, and predictive insights.",
        deployment: "On-premise deployment with secure data handling and HIPAA compliance."
      },
      results: [
        "Reduced average patient waiting time by 47%",
        "Improved staff utilization efficiency by 62%",
        "Decreased emergency department overcrowding by 38%",
        "Optimized inventory management, reducing waste by 25%"
      ],
      github: "https://github.com/henrymuli/healthcare-optimization",
      demo: "https://knh-analytics-demo.streamlit.app",
      documentation: "Implementation guide, data privacy protocols, and user manuals for hospital staff"
    }
  ];

  return (
    <section id="projects" className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Complete Data Science Projects & Case Studies
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Production-ready data science solutions with complete source code, documentation, and deployment guides. Each project includes real-world impact metrics and is available on GitHub.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {projects.map((project, index) => (
            <Card key={index} className="overflow-hidden hover:shadow-xl transition-all duration-300 hover:-translate-y-1 border-0 shadow-lg cursor-pointer">
              <div className="relative">
                <img 
                  src={project.image} 
                  alt={project.title}
                  className="w-full h-48 object-cover"
                />
                <div className="absolute top-4 left-4">
                  <Badge variant="secondary" className="bg-white/90 text-gray-800 font-medium">
                    {project.category}
                  </Badge>
                </div>
                <div className="absolute top-4 right-4">
                  <Badge className="bg-blue-600 text-white">
                    Production Ready
                  </Badge>
                </div>
              </div>
              
              <CardHeader className="pb-4">
                <CardTitle className="text-xl font-bold text-gray-900 leading-tight">
                  {project.title}
                </CardTitle>
                <CardDescription className="text-gray-600 leading-relaxed">
                  {project.description}
                </CardDescription>
              </CardHeader>
              
              <CardContent className="space-y-6">
                <div className="flex flex-wrap gap-2">
                  {project.tags.map((tag, tagIndex) => (
                    <Badge key={tagIndex} variant="outline" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                </div>
                
                <div className="grid grid-cols-3 gap-4">
                  {project.metrics.map((metric, metricIndex) => (
                    <div key={metricIndex} className="text-center p-3 bg-gray-50 rounded-lg">
                      <metric.icon className="h-5 w-5 text-blue-600 mx-auto mb-1" />
                      <div className="font-bold text-gray-900 text-sm">{metric.value}</div>
                      <div className="text-xs text-gray-600">{metric.label}</div>
                    </div>
                  ))}
                </div>

                <div className="flex gap-2">
                  <Button asChild size="sm" className="flex-1">
                    <a href={project.github} target="_blank" rel="noopener noreferrer">
                      <Github className="mr-2 h-4 w-4" />
                      View Code
                    </a>
                  </Button>
                  <Button variant="outline" asChild size="sm" className="flex-1">
                    <a href={project.demo} target="_blank" rel="noopener noreferrer">
                      <Eye className="mr-2 h-4 w-4" />
                      Live Demo
                    </a>
                  </Button>
                </div>
                
                <Dialog>
                  <DialogTrigger asChild>
                    <Button variant="outline" className="w-full group">
                      View Complete Case Study
                      <ArrowRight className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-transform" />
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="max-w-6xl max-h-[90vh] overflow-y-auto">
                    <DialogHeader>
                      <DialogTitle className="text-2xl font-bold text-gray-900 mb-2">
                        {project.title}
                      </DialogTitle>
                      <div className="flex gap-2 mb-4">
                        <Badge className="w-fit">{project.category}</Badge>
                        <Badge variant="outline" className="w-fit">Production Ready</Badge>
                      </div>
                    </DialogHeader>
                    
                    <div className="space-y-8">
                      <img 
                        src={project.image} 
                        alt={project.title}
                        className="w-full h-64 object-cover rounded-lg"
                      />
                      
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div className="md:col-span-2">
                          <h4 className="font-semibold text-xl mb-4">Project Overview</h4>
                          <DialogDescription className="text-base text-gray-700 leading-relaxed mb-6">
                            {project.detailedDescription}
                          </DialogDescription>
                          
                          <h4 className="font-semibold text-lg mb-3">Key Features & Implementation</h4>
                          <ul className="space-y-2 mb-6">
                            {project.keyFeatures.map((feature, idx) => (
                              <li key={idx} className="flex items-start">
                                <CheckCircle className="h-4 w-4 text-green-600 mt-0.5 mr-2 flex-shrink-0" />
                                <span className="text-gray-700">{feature}</span>
                              </li>
                            ))}
                          </ul>

                          {project.implementation && (
                            <>
                              <h4 className="font-semibold text-lg mb-3">Technical Implementation</h4>
                              <div className="space-y-3 mb-6">
                                {Object.entries(project.implementation).map(([key, value], idx) => (
                                  <div key={idx} className="border-l-4 border-blue-200 pl-4">
                                    <h5 className="font-medium text-gray-900 capitalize">{key.replace(/([A-Z])/g, ' $1').trim()}</h5>
                                    <p className="text-gray-600 text-sm">{value}</p>
                                  </div>
                                ))}
                              </div>
                            </>
                          )}

                          {project.results && (
                            <>
                              <h4 className="font-semibold text-lg mb-3">Results & Impact</h4>
                              <ul className="space-y-2 mb-6">
                                {project.results.map((result, idx) => (
                                  <li key={idx} className="flex items-start">
                                    <TrendingUp className="h-4 w-4 text-blue-600 mt-0.5 mr-2 flex-shrink-0" />
                                    <span className="text-gray-700">{result}</span>
                                  </li>
                                ))}
                              </ul>
                            </>
                          )}
                        </div>

                        <div className="space-y-6">
                          <div>
                            <h4 className="font-semibold text-lg mb-3">Technologies Used</h4>
                            <div className="flex flex-wrap gap-2">
                              {project.technologies.map((tech, idx) => (
                                <Badge key={idx} variant="secondary" className="text-sm">
                                  {tech}
                                </Badge>
                              ))}
                            </div>
                          </div>
                          
                          <div>
                            <h4 className="font-semibold text-lg mb-3">Project Metrics</h4>
                            <div className="space-y-3">
                              {project.metrics.map((metric, idx) => (
                                <div key={idx} className="text-center p-4 bg-blue-50 rounded-lg">
                                  <metric.icon className="h-6 w-6 text-blue-600 mx-auto mb-2" />
                                  <div className="font-bold text-blue-900 text-lg">{metric.value}</div>
                                  <div className="text-sm text-blue-700">{metric.label}</div>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex gap-4 pt-6 border-t">
                        <Button asChild className="flex-1">
                          <a href={project.github} target="_blank" rel="noopener noreferrer">
                            <Github className="mr-2 h-4 w-4" />
                            Complete Source Code
                          </a>
                        </Button>
                        <Button variant="outline" asChild className="flex-1">
                          <a href={project.demo} target="_blank" rel="noopener noreferrer">
                            <Eye className="mr-2 h-4 w-4" />
                            Live Demo & Documentation
                          </a>
                        </Button>
                      </div>
                    </div>
                  </DialogContent>
                </Dialog>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="mt-16 text-center">
          <Button size="lg" className="bg-blue-600 hover:bg-blue-700">
            <Github className="mr-2 h-5 w-5" />
            View All Projects on GitHub
          </Button>
        </div>
      </div>
    </section>
  );
};

export default Projects;
