
import React from 'react';
import { 
  FileSpreadsheet, 
  Zap, 
  Database, 
  BarChart3, 
  Settings, 
  MessageSquare,
  Mail,
  Workflow,
  Code,
  Globe
} from 'lucide-react';

const Skills = () => {
  const skills = [
    {
      name: 'Microsoft Excel',
      icon: FileSpreadsheet,
      description: 'Advanced formulas, pivot tables, VBA macros',
      color: 'bg-green-100 text-green-600',
      level: 95
    },
    {
      name: 'Power Automate',
      icon: Zap,
      description: 'Workflow design, triggers, complex automations',
      color: 'bg-blue-100 text-blue-600',
      level: 90
    },
    {
      name: 'SharePoint',
      icon: Database,
      description: 'Document libraries, list automation, workflows',
      color: 'bg-purple-100 text-purple-600',
      level: 85
    },
    {
      name: 'Power BI',
      icon: BarChart3,
      description: 'Dynamic dashboards, data visualization, reporting',
      color: 'bg-yellow-100 text-yellow-600',
      level: 80
    },
    {
      name: 'API Integration',
      icon: Settings,
      description: 'REST APIs, webhooks, third-party connections',
      color: 'bg-indigo-100 text-indigo-600',
      level: 85
    },
    {
      name: 'WhatsApp Business API',
      icon: MessageSquare,
      description: 'Automated messaging, customer communication',
      color: 'bg-green-100 text-green-600',
      level: 75
    },
    {
      name: 'Email Automation',
      icon: Mail,
      description: 'SMTP, Outlook integration, automated campaigns',
      color: 'bg-red-100 text-red-600',
      level: 90
    },
    {
      name: 'Zapier',
      icon: Workflow,
      description: 'Cross-platform automation, trigger workflows',
      color: 'bg-orange-100 text-orange-600',
      level: 80
    },
    {
      name: 'Python',
      icon: Code,
      description: 'Data analysis, scripting, automation scripts',
      color: 'bg-blue-100 text-blue-600',
      level: 70
    },
    {
      name: 'Web APIs',
      icon: Globe,
      description: 'JSON, XML, HTTP protocols, data exchange',
      color: 'bg-teal-100 text-teal-600',
      level: 75
    }
  ];

  return (
    <section id="skills" className="py-20 bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Skills & Technologies
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            A comprehensive toolkit for automating workflows, analyzing data, and integrating systems 
            to drive business efficiency and growth.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {skills.map((skill, index) => (
            <div key={index} className="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 hover:-translate-y-1">
              <div className="flex items-center gap-4 mb-4">
                <div className={`p-3 rounded-lg ${skill.color}`}>
                  <skill.icon className="h-6 w-6" />
                </div>
                <div className="flex-1">
                  <h3 className="font-semibold text-gray-900 text-lg">{skill.name}</h3>
                  <div className="flex items-center gap-2 mt-1">
                    <div className="flex-1 bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-blue-600 h-2 rounded-full transition-all duration-1000"
                        style={{ width: `${skill.level}%` }}
                      ></div>
                    </div>
                    <span className="text-sm text-gray-600 font-medium">{skill.level}%</span>
                  </div>
                </div>
              </div>
              <p className="text-gray-600 text-sm leading-relaxed">{skill.description}</p>
            </div>
          ))}
        </div>

        <div className="mt-16 text-center">
          <div className="inline-flex items-center gap-8 bg-white rounded-2xl px-8 py-6 shadow-lg">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">10+</div>
              <div className="text-sm text-gray-600">Technologies Mastered</div>
            </div>
            <div className="w-px h-12 bg-gray-200"></div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">500+</div>
              <div className="text-sm text-gray-600">Hours Automated</div>
            </div>
            <div className="w-px h-12 bg-gray-200"></div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">99%</div>
              <div className="text-sm text-gray-600">Uptime Achieved</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Skills;
