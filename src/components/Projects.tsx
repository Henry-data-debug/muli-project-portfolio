
import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { ExternalLink, TrendingUp, Users, Clock, CheckCircle, ArrowRight, Github, Eye } from 'lucide-react';

const Projects = () => {
  const [selectedProject, setSelectedProject] = useState(null);

  const projects = [
    {
      title: "Customer Churn Prediction Model",
      description: "Built a machine learning model using Python and scikit-learn to predict customer churn for a Kenyan telecom company, achieving 94% accuracy and helping reduce churn by 23%.",
      image: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=600&h=400&fit=crop",
      tags: ["Python", "Scikit-learn", "Pandas", "Random Forest"],
      metrics: [
        { label: "Model Accuracy", value: "94%", icon: CheckCircle },
        { label: "Churn Reduction", value: "23%", icon: TrendingUp },
        { label: "Customers Analyzed", value: "50K+", icon: Users }
      ],
      category: "Machine Learning",
      detailedDescription: "Developed a comprehensive customer churn prediction system for Safaricom Kenya. The project involved data preprocessing of customer usage patterns, demographic information, and billing history. Implemented feature engineering to create meaningful predictors and used ensemble methods including Random Forest and Gradient Boosting.",
      keyFeatures: [
        "Real-time churn risk scoring for 50,000+ customers",
        "Feature importance analysis identifying top churn drivers",
        "Automated monthly model retraining pipeline",
        "Interactive dashboard for business stakeholders",
        "A/B testing framework for retention strategies"
      ],
      technologies: ["Python", "Pandas", "Scikit-learn", "XGBoost", "Matplotlib", "Seaborn", "SQL", "Apache Airflow"],
      github: "https://github.com/henrymuli/churn-prediction",
      demo: "https://churn-dashboard.herokuapp.com"
    },
    {
      title: "Sales Forecasting Dashboard",
      description: "Created an advanced time series forecasting model using ARIMA and Prophet to predict quarterly sales for retail chains across Kenya, improving forecast accuracy by 40%.",
      image: "https://images.unsplash.com/photo-1460925895917-afdab827c52f?q=80&w=600&h=400&fit=crop",
      tags: ["Python", "Prophet", "Time Series", "Power BI"],
      metrics: [
        { label: "Forecast Accuracy", value: "87%", icon: CheckCircle },
        { label: "Improvement", value: "40%", icon: TrendingUp },
        { label: "Retail Stores", value: "150+", icon: Users }
      ],
      category: "Time Series Analysis",
      detailedDescription: "Built an end-to-end sales forecasting solution for Nakumatt Holdings (before closure) and other major Kenyan retail chains. The system processes historical sales data, external factors like holidays and weather, and economic indicators to generate accurate quarterly forecasts.",
      keyFeatures: [
        "Multi-variate time series modeling with external regressors",
        "Seasonal decomposition and trend analysis",
        "Confidence intervals and uncertainty quantification",
        "What-if scenario analysis for business planning",
        "Automated report generation and email alerts"
      ],
      technologies: ["Python", "Prophet", "ARIMA", "Pandas", "NumPy", "Power BI", "SQL Server", "Azure"],
      github: "https://github.com/henrymuli/sales-forecasting",
      demo: "https://sales-forecast-demo.herokuapp.com"
    },
    {
      title: "Fraud Detection System",
      description: "Designed and implemented a real-time fraud detection system for M-Pesa transactions using anomaly detection algorithms, reducing fraudulent transactions by 78%.",
      image: "https://images.unsplash.com/photo-1563013544-824ae1b704d3?q=80&w=600&h=400&fit=crop",
      tags: ["Python", "Anomaly Detection", "Real-time", "Apache Kafka"],
      metrics: [
        { label: "Fraud Reduction", value: "78%", icon: CheckCircle },
        { label: "Processing Speed", value: "<100ms", icon: Clock },
        { label: "Daily Transactions", value: "1M+", icon: TrendingUp }
      ],
      category: "Anomaly Detection",
      detailedDescription: "Developed a sophisticated fraud detection system for Safaricom's M-Pesa platform. The system uses unsupervised learning algorithms to identify suspicious transaction patterns in real-time, processing over 1 million transactions daily.",
      keyFeatures: [
        "Real-time transaction scoring using Isolation Forest",
        "Network analysis to detect fraud rings",
        "Dynamic threshold adjustment based on transaction patterns",
        "False positive reduction through ensemble methods",
        "Integration with existing fraud investigation workflows"
      ],
      technologies: ["Python", "Scikit-learn", "Apache Kafka", "Redis", "PostgreSQL", "Docker", "Kubernetes"],
      github: "https://github.com/henrymuli/fraud-detection",
      demo: "https://fraud-detection-demo.herokuapp.com"
    },
    {
      title: "Agricultural Yield Prediction",
      description: "Developed a crop yield prediction model using satellite imagery and weather data for smallholder farmers in Kenya, helping optimize planting decisions and increase yields by 25%.",
      image: "https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?q=80&w=600&h=400&fit=crop",
      tags: ["Python", "Computer Vision", "TensorFlow", "Satellite Data"],
      metrics: [
        { label: "Yield Increase", value: "25%", icon: TrendingUp },
        { label: "Farmers Helped", value: "5000+", icon: Users },
        { label: "Prediction Accuracy", value: "91%", icon: CheckCircle }
      ],
      category: "Computer Vision",
      detailedDescription: "Created an AI-powered agricultural advisory system for the Kenya Agricultural Research Institute (KALRO). The system combines satellite imagery analysis, weather forecasting, and soil data to provide personalized crop recommendations to farmers.",
      keyFeatures: [
        "Satellite image analysis using convolutional neural networks",
        "Weather pattern recognition and forecasting",
        "Soil moisture and nutrient level prediction",
        "Mobile app for farmer recommendations",
        "Integration with local weather stations"
      ],
      technologies: ["Python", "TensorFlow", "OpenCV", "Google Earth Engine", "Flask", "React Native"],
      github: "https://github.com/henrymuli/crop-prediction",
      demo: "https://crop-advisor-demo.herokuapp.com"
    },
    {
      title: "Healthcare Analytics Platform",
      description: "Built a comprehensive healthcare analytics platform for Kenyatta National Hospital to optimize patient flow, reduce waiting times, and improve resource allocation using predictive modeling.",
      image: "https://images.unsplash.com/photo-1576091160399-112ba8d25d1f?q=80&w=600&h=400&fit=crop",
      tags: ["Python", "Healthcare", "Predictive Analytics", "Streamlit"],
      metrics: [
        { label: "Wait Time Reduction", value: "45%", icon: Clock },
        { label: "Resource Efficiency", value: "60%", icon: TrendingUp },
        { label: "Patients Served", value: "10K+", icon: Users }
      ],
      category: "Healthcare Analytics",
      detailedDescription: "Developed a data-driven healthcare management system for Kenya's largest referral hospital. The platform analyzes patient admission patterns, staff scheduling, and resource utilization to optimize hospital operations.",
      keyFeatures: [
        "Patient admission forecasting using time series analysis",
        "Staff scheduling optimization algorithms",
        "Resource allocation recommendations",
        "Real-time dashboard for hospital administrators",
        "Patient flow visualization and bottleneck identification"
      ],
      technologies: ["Python", "Streamlit", "Pandas", "Scikit-learn", "PostgreSQL", "Plotly", "Apache Superset"],
      github: "https://github.com/henrymuli/healthcare-analytics",
      demo: "https://healthcare-dashboard-demo.herokuapp.com"
    },
    {
      title: "Market Basket Analysis System",
      description: "Implemented advanced market basket analysis for Carrefour Kenya to optimize product placement and cross-selling strategies, increasing average transaction value by 18%.",
      image: "https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?q=80&w=600&h=400&fit=crop",
      tags: ["Python", "Association Rules", "Retail Analytics", "Apriori"],
      metrics: [
        { label: "Transaction Value", value: "+18%", icon: TrendingUp },
        { label: "Cross-sell Rate", value: "+32%", icon: CheckCircle },
        { label: "Products Analyzed", value: "10K+", icon: Users }
      ],
      category: "Retail Analytics",
      detailedDescription: "Created a sophisticated market basket analysis system for one of Kenya's largest retail chains. The system identifies purchasing patterns, product associations, and customer segments to drive strategic merchandising decisions.",
      keyFeatures: [
        "Apriori algorithm implementation for frequent itemset mining",
        "Customer segmentation using clustering algorithms",
        "Dynamic pricing recommendations",
        "Product placement optimization",
        "Seasonal trend analysis and inventory planning"
      ],
      technologies: ["Python", "Pandas", "MLxtend", "Scikit-learn", "Plotly", "Streamlit", "BigQuery"],
      github: "https://github.com/henrymuli/market-basket-analysis",
      demo: "https://market-analysis-demo.herokuapp.com"
    }
  ];

  return (
    <section id="projects" className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Data Science Projects & Case Studies
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Real-world data science solutions that deliver measurable business impact across various industries in Kenya.
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
                
                <Dialog>
                  <DialogTrigger asChild>
                    <Button variant="outline" className="w-full group">
                      View Case Study Details
                      <ArrowRight className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-transform" />
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
                    <DialogHeader>
                      <DialogTitle className="text-2xl font-bold text-gray-900 mb-2">
                        {project.title}
                      </DialogTitle>
                      <Badge className="w-fit mb-4">{project.category}</Badge>
                    </DialogHeader>
                    
                    <div className="space-y-6">
                      <img 
                        src={project.image} 
                        alt={project.title}
                        className="w-full h-64 object-cover rounded-lg"
                      />
                      
                      <DialogDescription className="text-base text-gray-700 leading-relaxed">
                        {project.detailedDescription}
                      </DialogDescription>
                      
                      <div>
                        <h4 className="font-semibold text-lg mb-3">Key Features & Implementation</h4>
                        <ul className="space-y-2">
                          {project.keyFeatures.map((feature, idx) => (
                            <li key={idx} className="flex items-start">
                              <CheckCircle className="h-4 w-4 text-green-600 mt-0.5 mr-2 flex-shrink-0" />
                              <span className="text-gray-700">{feature}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                      
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
                        <h4 className="font-semibold text-lg mb-3">Project Impact</h4>
                        <div className="grid grid-cols-3 gap-4">
                          {project.metrics.map((metric, idx) => (
                            <div key={idx} className="text-center p-4 bg-blue-50 rounded-lg">
                              <metric.icon className="h-6 w-6 text-blue-600 mx-auto mb-2" />
                              <div className="font-bold text-blue-900 text-lg">{metric.value}</div>
                              <div className="text-sm text-blue-700">{metric.label}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                      
                      <div className="flex gap-4 pt-4">
                        <Button asChild className="flex-1">
                          <a href={project.github} target="_blank" rel="noopener noreferrer">
                            <Github className="mr-2 h-4 w-4" />
                            View on GitHub
                          </a>
                        </Button>
                        <Button variant="outline" asChild className="flex-1">
                          <a href={project.demo} target="_blank" rel="noopener noreferrer">
                            <Eye className="mr-2 h-4 w-4" />
                            Live Demo
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
