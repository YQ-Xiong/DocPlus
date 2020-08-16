/**
 * 
 */
package structures;
import java.util.*;
import java.util.HashMap;
import java.io.Serializable;
import json.JSONException;
import json.JSONObject;

/**
 * @author Yingqiao Xiong
 * @category data structure
 * data structure for a Yelp review document
 * You can create some necessary data structure here to store the processed text content, e.g., bag-of-word representation
 */
public class Post implements Serializable{
	//unique review ID from Yelp
	String m_ID;		
	public void setID(String ID) {
		m_ID = ID;
	}
	
	public String getID() {
		return m_ID;
	}

	//author's displayed name
	String m_author;	
	public String getAuthor() {
		return m_author;
	}

	public void setAuthor(String author) {
		this.m_author = author;
	}
	
	//author's location
	String m_location;
	public String getLocation() {
		return m_location;
	}

	public void setLocation(String location) {
		this.m_location = location;
	}

	//review text content
	String m_content;
	public String getContent() {
		return m_content;
	}

	public void setContent(String content) {
		if (!content.isEmpty())
			this.m_content = content;
	}
	
	public boolean isEmpty() {
		return m_content==null || m_content.isEmpty();
	}

	//timestamp of the post
	String m_date;
	public String getDate() {
		return m_date;
	}

	public void setDate(String date) {
		this.m_date = date;
	}
	
	//overall rating to the business in this review
	double m_rating;
	public double getRating() {
		return m_rating;
	}

	public void setRating(double rating) {
		this.m_rating = rating;
	}

	public Post(String ID) {
		m_ID = ID;
	}
	
	String[] m_tokens; // we will store the tokens
	public String[] getTokens() {
		return m_tokens;
	}
	
	public void setTokens(String[] tokens) {
		m_tokens = tokens;
	}

	Set<String> m_vocabs;
	public Set<String> getVocabs(){return m_vocabs;}

	public void setVocab(Set<String> vocab){m_vocabs = vocab;}
	
	HashMap<String, Double> m_vector; // suggested sparse structure for storing the vector space representation with N-grams for this document
	public HashMap<String, Double> getVct() {
		return m_vector;
	}
	
	public void setVct(HashMap<String, Double> vct) {
		m_vector = vct;
	}

	int m_label;
	public void setLabel(int l){m_label = l;}
	public int getLabel(){return m_label;}


	public double similiarity(Post p) {
		HashMap<String, Double> p_vector = p.getVct();
		double m_norm = 0.0;
		double p_norm = 0.0;
		double sim = 0.0;

		for(String key : m_vector.keySet()) {
            m_norm += m_vector.get(key) * m_vector.get(key);
        }
        for(String key : p_vector.keySet()) {
            p_norm += p_vector.get(key) * p_vector.get(key);
        }
        m_norm = Math.sqrt(m_norm);
		p_norm = Math.sqrt(p_norm);

		for(String key : m_vector.keySet()){
		    if(p_vector.containsKey(key)){
		        sim += m_vector.get(key) * p_vector.get(key);
            }

        }

		return m_norm == 0.0 || p_norm == 0.0 ? 0.0 : sim / (m_norm * p_norm);

	}
	
	public Post(JSONObject json) {
		try {
			m_ID = json.getString("ReviewID");
			setAuthor(json.getString("Author"));
			
			setDate(json.getString("Date"));			
			setContent(json.getString("Content"));
			setRating(json.getDouble("Overall"));
			setLocation(json.getString("Author_Location"));
			setLabel(json.getDouble("Overall") >=4 ? 1 : 0);
		} catch (JSONException e) {
			e.printStackTrace();
		}
	}
	
	public JSONObject getJSON() throws JSONException {
		JSONObject json = new JSONObject();
		
		json.put("ReviewID", m_ID);//must contain
		json.put("Author", m_author);//must contain
		json.put("Date", m_date);//must contain
		json.put("Content", m_content);//must contain
		json.put("Overall", m_rating);//must contain
		json.put("Author_Location", m_location);//must contain
		
		return json;
	}
}
