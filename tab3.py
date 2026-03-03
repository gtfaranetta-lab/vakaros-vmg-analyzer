        with tab3:
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            scatter = ax3.scatter(df['latitude'], df['longitude'], 
                                c=df['VMG'], cmap='RdYlGn', 
                                s=30, alpha=0.8)
            ax3.plot(waypoint_lat, waypoint_lon, 'r*', markersize=20, 
                    label='Waypoint', markeredgecolor='black', markeredgewidth=1)
            ax3.set_ylim(df['longitude'].max(), df['longitude'].min())
            ax3.set_xlabel('Latitude')
            ax3.set_ylabel('Longitude')
            ax3.set_title('Track (colored by VMG)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.axis('equal')
            plt.colorbar(scatter, ax=ax3, label='VMG (knots)')
            st.pyplot(fig3)
