
import React from 'react';
import { CheckCircle, Target, Lightbulb, Users } from 'lucide-react';

const About = () => {
  return (
    <section id="about" className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          <div className="space-y-6">
            <div>
              <h2 className="text-4xl font-bold text-gray-900 mb-4">
                About Me
              </h2>
              <p className="text-lg text-gray-600 leading-relaxed">
                I'm a passionate Data Scientist and Workflow Automation Specialist with over 5 years of experience 
                transforming business processes through intelligent automation and data-driven solutions.
              </p>
            </div>
            
            <p className="text-gray-600 leading-relaxed">
              My journey began in traditional data analysis, but I quickly discovered my passion for automation 
              when I saw how much time businesses waste on repetitive tasks. Today, I specialize in creating 
              seamless workflows that not only save time but also eliminate human error and improve accuracy.
            </p>
            
            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <CheckCircle className="h-6 w-6 text-green-600 mt-0.5 flex-shrink-0" />
                <div>
                  <h4 className="font-semibold text-gray-900">Client-First Mindset</h4>
                  <p className="text-gray-600">Every solution is tailored to your specific business needs and goals.</p>
                </div>
              </div>
              
              <div className="flex items-start gap-3">
                <Target className="h-6 w-6 text-blue-600 mt-0.5 flex-shrink-0" />
                <div>
                  <h4 className="font-semibold text-gray-900">Results-Driven Approach</h4>
                  <p className="text-gray-600">Focus on measurable outcomes and continuous improvement.</p>
                </div>
              </div>
              
              <div className="flex items-start gap-3">
                <Lightbulb className="h-6 w-6 text-yellow-600 mt-0.5 flex-shrink-0" />
                <div>
                  <h4 className="font-semibold text-gray-900">Innovation & Efficiency</h4>
                  <p className="text-gray-600">Always exploring new ways to optimize and streamline processes.</p>
                </div>
              </div>
            </div>
          </div>
          
          <div className="relative">
            <img 
              src="https://images.unsplash.com/photo-1498050108023-c5249f4df085?q=80&w=800&h=600&fit=crop"
              alt="Data analysis and workflow automation"
              className="rounded-2xl shadow-lg w-full h-96 object-cover"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent rounded-2xl"></div>
          </div>
        </div>
        
        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="text-center p-6 bg-gray-50 rounded-xl">
            <Users className="h-12 w-12 text-blue-600 mx-auto mb-4" />
            <h3 className="text-2xl font-bold text-gray-900 mb-2">100+</h3>
            <p className="text-gray-600">Happy Clients</p>
          </div>
          
          <div className="text-center p-6 bg-gray-50 rounded-xl">
            <Target className="h-12 w-12 text-green-600 mx-auto mb-4" />
            <h3 className="text-2xl font-bold text-gray-900 mb-2">200+</h3>
            <p className="text-gray-600">Projects Completed</p>
          </div>
          
          <div className="text-center p-6 bg-gray-50 rounded-xl">
            <Lightbulb className="h-12 w-12 text-purple-600 mx-auto mb-4" />
            <h3 className="text-2xl font-bold text-gray-900 mb-2">5+</h3>
            <p className="text-gray-600">Years Experience</p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default About;
